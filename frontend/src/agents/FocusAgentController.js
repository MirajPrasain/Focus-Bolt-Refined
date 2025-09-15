import { detectFaceLandmarks } from "../detectors/faceMeshDetector";
import { detectPhone } from "../detectors/phoneDetector";
import { isBlinking } from "../detectors/blinkDetector";
import { isLookingAtScreen } from "../detectors/irisFocusDetector";
import { getHeadTilt } from "../detectors/headTiltDetector";

export async function processFrame({
  faceModel,
  phoneModel,
  video,
  onResults
}) {
  const faces = await detectFaceLandmarks(faceModel, video);
  const phoneDetected = await detectPhone(phoneModel, video);

  let blink = false, looking = false, tilt = 0, multiFace = false;
  if (faces.length > 0) {
    const landmarks = faces[0].scaledMesh;
    blink = isBlinking(landmarks);
    looking = isLookingAtScreen(landmarks);
    tilt = getHeadTilt(landmarks);
    multiFace = faces.length > 1;
  }

  onResults({
    blink,
    looking,
    tilt,
    phoneDetected,
    multiFace
  });
} 