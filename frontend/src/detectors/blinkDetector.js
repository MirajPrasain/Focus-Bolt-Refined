export function isBlinking(landmarks, threshold = 0.2) {
  const leftEye = [33, 160, 158, 133, 153, 144];
  const rightEye = [362, 385, 387, 263, 373, 380];

  function eyeAspectRatio(eye) {
    const [p1, p2, p3, p4, p5, p6] = eye.map(i => landmarks[i]);
    const distV1 = Math.hypot(p2.x - p6.x, p2.y - p6.y);
    const distV2 = Math.hypot(p3.x - p5.x, p3.y - p5.y);
    const distH = Math.hypot(p1.x - p4.x, p1.y - p4.y);
    return (distV1 + distV2) / (2.0 * distH);
  }

  const leftEAR = eyeAspectRatio(leftEye);
  const rightEAR = eyeAspectRatio(rightEye);
  return (leftEAR + rightEAR) / 2 < threshold;
} 