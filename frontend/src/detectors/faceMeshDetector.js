import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";

export async function loadFaceMeshModel() {
  return await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh
  );
}

export async function detectFaceLandmarks(model, video) {
  const predictions = await model.estimateFaces({ input: video });
  return predictions;
} 