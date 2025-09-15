export function getHeadTilt(landmarks) {
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);
  return angle;
} 