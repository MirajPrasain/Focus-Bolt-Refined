"""
FocusAgent — study_mode.py (Apple‑grade smoothing)
=================================================

Design goals (Apple‑like):
• Temporal smoothing of raw face/pose/iris signals (One‑Euro + EMA)
• Momentum‑based score dynamics (graceful decay & recovery)
• Contextual stability (debounce / streak gating / hysteresis)
• Soft penalties (sigmoid ramps) instead of harsh steps
• Perceptual mapping (gamma) for calm UI feel
• Event hysteresis for phone / head‑down / turn / multi‑face
• Clean, testable architecture with single entry: process_frame()

Drop‑in: Keeps your public functions and event arrays compatible with existing backend.
"""

from __future__ import annotations
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# =====================
# Mediapipe & YOLO init
# =====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

model = YOLO("yolov5s.pt")

# =====================
# Session state
# =====================
focus_scores: list = []       # smoothed, perceptual scores (0..100)
cheat_times: list = []        # timestamps of events
cheat_event: list = []        # 1=down, 2=turn, 3=multi-face, 4=phone

phone_checks = 0
face_events = 0
turn_events = 0
down_events = 0

phone_flag = False
face_flag = False
turn_flag = False
down_flag = False

blink_counter = 0
start_time = time.time()
SESSION_DURATION = 30  # seconds, configurable via set_session_duration()

# =====================
# Config
# =====================
@dataclass
class FocusConfig:
    # Smoothing
    ema_alpha: float = 0.12                 # EMA smoothing for final score (0=stiff, 1=reactive)
    gamma: float = 0.65                     # perceptual mapping exponent (0.4‑0.8 feels calm)

    # Momentum
    decay_rate: float = 0.92                # when unfocused, blend down over time
    recovery_rate: float = 0.12             # when focused, ease back up

    # Stability / Debounce
    loss_tolerance_frames: int = 4          # brief dips ignored (2‑6 typical)
    event_on_streak: int = 4                # frames required to register event
    event_off_streak: int = 6               # frames clear to reset the event
    blink_ea_threshold: float = 0.20        # eye aspect threshold for blink
    blink_consec_required: int = 3

    # Soft penalty ramps (centers & softness)
    iris_center: float = 0.50
    iris_soft: float = 0.07                 # larger = softer rolloff
    iris_allow: float = 0.18                # allowable range around center

    vert_center: float = 0.50
    vert_soft: float = 0.08
    vert_allow: float = 0.18

    tilt_center: float = 1.0                # left/right temple distance ratio (≈1 when straight)
    tilt_soft: float = 0.18
    tilt_allow: float = 0.35

    down_center: float = 1.0                # nose‑chin vs eye‑nose ratio ≈1 when neutral
    down_soft: float = 0.16
    down_allow: float = 0.35

    # Weights
    w_iris: float = 0.35
    w_vert: float = 0.20
    w_tilt: float = 0.25
    w_down: float = 0.20

    # Hard overrides
    phone_zero_score: bool = True
    multi_face_penalty: float = 0.30        # reduce composite by 30%

CFG = FocusConfig()

# =====================
# Utilities
# =====================
def euclidean(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# ---------- One‑Euro filter for raw landmark scalars ----------
class OneEuroFilter:
    def __init__(self, freq=60.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.last_time = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, t, x):
        if self.last_time is None:
            self.last_time = t
        dt = max(t - self.last_time, 1e-6)
        self.freq = 1.0 / dt
        self.last_time = t

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

# filters for key scalars
f_iris_h = OneEuroFilter(min_cutoff=1.5, beta=0.02)
f_iris_v = OneEuroFilter(min_cutoff=1.5, beta=0.02)
f_tilt = OneEuroFilter(min_cutoff=1.2, beta=0.01)
f_down = OneEuroFilter(min_cutoff=1.2, beta=0.01)

# EMA for final score
smoothed_focus = None
last_focus_state = 100.0
loss_streak = 0

# Event debouncers
class Debounce:
    def __init__(self, on_streak: int, off_streak: int):
        self.on_streak = on_streak
        self.off_streak = off_streak
        self.count = 0
        self.active = False

    def update(self, active_now: bool) -> bool:
        if active_now:
            self.count = min(self.on_streak, self.count + 1)
            if self.count >= self.on_streak:
                self.active = True
        else:
            self.count = min(self.off_streak, self.count + 1) if self.active else 0
            if self.count >= self.off_streak:
                self.active = False
                self.count = 0
        return self.active

phone_db = Debounce(CFG.event_on_streak, CFG.event_off_streak)
turn_db = Debounce(CFG.event_on_streak, CFG.event_off_streak)
down_db = Debounce(CFG.event_on_streak, CFG.event_off_streak)
face_db = Debounce(CFG.event_on_streak, CFG.event_off_streak)

# =====================
# Landmark helpers
# =====================
def _get_point(lm, idx, w, h):
    p = lm[idx]
    return int(p.x * w), int(p.y * h)


def eye_openness(eye_top, eye_bottom, eye_left, eye_right):
    vertical_openness = euclidean(eye_top, eye_bottom)
    horizontal_openness = euclidean(eye_left, eye_right)
    return vertical_openness / (horizontal_openness + 1e-6)


def iris_position_ratio(iris_center, eye_left, eye_right, eye_top, eye_bottom):
    total_width = euclidean(eye_left, eye_right) + 1e-6
    total_height = euclidean(eye_top, eye_bottom) + 1e-6
    iris_to_left = euclidean(iris_center, eye_left)
    iris_to_top = euclidean(iris_center, eye_top)
    horizontal_ratio = iris_to_left / total_width
    vertical_ratio = iris_to_top / total_height
    return horizontal_ratio, vertical_ratio


def head_tilt_ratio(left_temple, right_temple, nose_tip):
    left_to_nose = euclidean(left_temple, nose_tip) + 1e-6
    right_to_nose = euclidean(right_temple, nose_tip) + 1e-6
    return left_to_nose / right_to_nose


def head_down_ratio(nose_tip, chin, eye_level):
    eye_to_nose = euclidean(eye_level, nose_tip) + 1e-6
    chin_to_nose = euclidean(nose_tip, chin) + 1e-6
    return eye_to_nose / chin_to_nose

# =====================
# Soft penalty helpers
# =====================
def soft_penalty(center: float, allow: float, soft: float, x: float) -> float:
    """Return 0..1 penalty using smooth logistic ramps outside [center±allow]."""
    left = center - allow
    right = center + allow
    if left <= x <= right:
        return 0.0
    # distance outside band
    d = (left - x) if x < left else (x - right)
    # logistic ramp (softer far away)
    return 1.0 / (1.0 + np.exp(-(d) / max(soft, 1e-6)))


def clamp01(x):
    return float(max(0.0, min(1.0, x)))

# =====================
# Core scoring
# =====================
def _raw_focus_components(iris_h, iris_v, tilt, down) -> Tuple[float, Dict[str, float]]:
    p_iris = soft_penalty(CFG.iris_center, CFG.iris_allow, CFG.iris_soft, iris_h)
    p_vert = soft_penalty(CFG.vert_center, CFG.vert_allow, CFG.vert_soft, iris_v)
    p_tilt = soft_penalty(CFG.tilt_center, CFG.tilt_allow, CFG.tilt_soft, tilt)
    p_down = soft_penalty(CFG.down_center, CFG.down_allow, CFG.down_soft, down)

    # weighted composite penalty 0..1
    total_pen = clamp01(
        CFG.w_iris * p_iris + CFG.w_vert * p_vert + CFG.w_tilt * p_tilt + CFG.w_down * p_down
    )
    # map to score 0..100
    base_score = (1.0 - total_pen) * 100.0
    return base_score, {
        "p_iris": p_iris,
        "p_vert": p_vert,
        "p_tilt": p_tilt,
        "p_down": p_down,
        "total_pen": total_pen,
    }


def _momentum_update(current_score: float) -> float:
    global last_focus_state
    if current_score < 50.0:
        last_focus_state = last_focus_state * CFG.decay_rate + current_score * (1 - CFG.decay_rate)
    else:
        last_focus_state = last_focus_state + CFG.recovery_rate * (current_score - last_focus_state)
    return float(max(0.0, min(100.0, last_focus_state)))


def _ema_update(x: float) -> float:
    global smoothed_focus
    if smoothed_focus is None:
        smoothed_focus = x
    else:
        a = clamp01(CFG.ema_alpha)
        smoothed_focus = a * x + (1 - a) * smoothed_focus
    return float(smoothed_focus)


def _perceptual_map(raw_0_100: float) -> float:
    x = clamp01(raw_0_100 / 100.0)
    return float((x ** CFG.gamma) * 100.0)

# =====================
# Public API: settings & getters
# =====================
def set_session_duration(seconds):
    global SESSION_DURATION, start_time
    SESSION_DURATION = int(seconds)
    start_time = time.time()


def get_session_duration():
    return SESSION_DURATION


def get_focus_data():
    return focus_scores


def get_cheat_data():
    return cheat_times, cheat_event

# =====================
# Detection helpers w/ hysteresis
# =====================
def _detect_phone(results_yolo, ts):
    global phone_checks, phone_flag
    phone_now = False
    for box in results_yolo.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        if label == "cell phone" and conf > 0.5:
            phone_now = True
            break
    active = phone_db.update(phone_now)
    if active and not phone_flag:
        phone_checks += 1
        phone_flag = True
        cheat_times.append(ts)
        cheat_event.append(4)
    if not active and phone_flag:
        phone_flag = False
    return active


def _detect_multiple_faces(result, ts):
    global face_events, face_flag
    multi_now = bool(result.multi_face_landmarks and len(result.multi_face_landmarks) > 1)
    active = face_db.update(multi_now)
    if active and not face_flag:
        face_events += 1
        face_flag = True
        cheat_times.append(ts)
        cheat_event.append(3)
    if not active and face_flag:
        face_flag = False
    return active


def _detect_head_pose(result, w, h, ts):
    global turn_events, down_events, turn_flag, down_flag
    if not result.multi_face_landmarks:
        turn_db.update(False)
        down_db.update(False)
        if turn_flag:
            turn_flag = False
        if down_flag:
            down_flag = False
        return False, False

    lm = result.multi_face_landmarks[0].landmark
    P = lambda i: _get_point(lm, i, w, h)
    nose = P(1)
    left_temple = P(234)
    right_temple = P(454)
    chin = P(152)
    eye_lvl = P(151)

    tilt = head_tilt_ratio(left_temple, right_temple, nose)
    down = head_down_ratio(nose, chin, eye_lvl)

    # dynamic smoothing for pose
    t = time.time()
    tilt_s = float(f_tilt(t, tilt))
    down_s = float(f_down(t, down))

    extreme_turn_now = (tilt_s > 1.5) or (tilt_s < 0.67)
    looking_down_now = (down_s > 1.4)

    turn_active = turn_db.update(extreme_turn_now)
    down_active = down_db.update(looking_down_now)

    if turn_active and not turn_flag:
        turn_events += 1
        turn_flag = True
        cheat_times.append(ts)
        cheat_event.append(2)
    if not turn_active and turn_flag:
        turn_flag = False

    if down_active and not down_flag:
        down_events += 1
        down_flag = True
        cheat_times.append(ts)
        cheat_event.append(1)
    if not down_active and down_flag:
        down_flag = False

    return turn_active, down_active

# =====================
# Core score from landmarks (soft penalties + stability)
# =====================
def _score_from_landmarks(result, w, h, phone_active):
    global blink_counter, loss_streak

    if not result.multi_face_landmarks:
        # No face: glide down but not instant zero
        return 20.0, "No face"

    lm = result.multi_face_landmarks[0].landmark
    P = lambda i: _get_point(lm, i, w, h)

    eye_top = P(159)
    eye_bottom = P(145)
    eye_left = P(33)
    eye_right = P(133)
    iris_center = P(468)
    nose_tip = P(1)
    left_temple = P(234)
    right_temple = P(454)
    chin = P(152)
    eye_level = P(151)

    ear = eye_openness(eye_top, eye_bottom, eye_left, eye_right)
    iris_h, iris_v = iris_position_ratio(iris_center, eye_left, eye_right, eye_top, eye_bottom)
    tilt = head_tilt_ratio(left_temple, right_temple, nose_tip)
    down = head_down_ratio(nose_tip, chin, eye_level)

    # Smooth raw ratios (sensor‑level calm)
    t = time.time()
    iris_h = float(f_iris_h(t, iris_h))
    iris_v = float(f_iris_v(t, iris_v))
    tilt = float(f_tilt(t, tilt))
    down = float(f_down(t, down))

    # Blink detection with streak gating
    if ear < CFG.blink_ea_threshold:
        blink_counter += 1
    else:
        blink_counter = 0

    if blink_counter >= CFG.blink_consec_required:
        # treat as temporary occlusion; do not crater score
        base = 35.0
        status = "Blink"
    else:
        base, comps = _raw_focus_components(iris_h, iris_v, tilt, down)
        status = "Focused" if base >= 60 else "Unstable"

    # Hard override: phone in view
    if CFG.phone_zero_score and phone_active:
        base = 0.0
        status = "Phone"

    # Stability gate: ignore brief dips < threshold for a few frames
    if base < 40.0:
        loss_streak += 1
    else:
        loss_streak = max(0, loss_streak - 1)

    if loss_streak < CFG.loss_tolerance_frames:
        # hold state by biasing toward last smoothed state (handled in momentum)
        base = max(base, last_focus_state - 5)

    return base, status

# =====================
# Main entry: process_frame
# =====================
def process_frame(frame, timestamp=None):
    """Process a BGR frame; returns JSON: score (0..100), cheat_events (latest), status, debug."""
    global focus_scores

    ts = (time.time() - start_time) if timestamp is None else float(timestamp)
    if ts > SESSION_DURATION:
        return "Session Ended"

    # YOLO phone detection on *original* frame (not flipped)
    results_yolo = model(frame)[0]
    phone_active = _detect_phone(results_yolo, ts)

    # Face mesh on mirrored frame for natural UX
    frame_flipped = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    h, w, _ = frame_flipped.shape

    # Other events
    _detect_multiple_faces(result, ts)
    _detect_head_pose(result, w, h, ts)

    # Base raw score from soft components + stability
    base, status = _score_from_landmarks(result, w, h, phone_active)

    # Momentum dynamics
    with_momentum = _momentum_update(base)

    # Perceptual mapping (gamma)
    perceptual = _perceptual_map(with_momentum)

    # EMA final smoothing for UI calm
    final_score = _ema_update(perceptual)

    # Multi‑face penalty (soft) after mapping
    if face_flag:
        final_score *= (1.0 - CFG.multi_face_penalty)

    # Clamp and record
    final_score = float(max(0.0, min(100.0, final_score)))

    # For analytics, only push when status suggests valid reading
    if status in ("Focused", "Unstable"):
        focus_scores.append(final_score)

    # Mark distract events if truly low (post‑smoothing), but avoid spamming
    if final_score < 35:
        cheat_times.append(round(ts, 2))

    payload = {
        "score": int(round(final_score)),
        "status": status,
        "cheat_events": cheat_event[-1:] if cheat_event else [],
        "debug": {
            "base": round(base, 2),
            "momentum": round(with_momentum, 2),
            "perceptual": round(perceptual, 2),
            "ema": round(final_score, 2),
            "phone": phone_active,
            "multi_face": face_flag,
            "turn": turn_flag,
            "down": down_flag,
            "loss_streak": loss_streak,
        },
    }
    return json.dumps(payload)

# =====================
# Optional: quick self‑test (synthetic)
# =====================
if __name__ == "__main__":
    # This block is a placeholder to sanity‑check imports/config.
    # Integrate with your WebSocket loop that feeds frames into process_frame().
    print("study_mode.py — loaded. Use process_frame(frame) in your streaming loop.")
