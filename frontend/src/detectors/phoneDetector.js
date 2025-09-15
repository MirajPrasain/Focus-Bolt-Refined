import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs-backend-webgl";

export async function loadPhoneModel() {
  return await cocoSsd.load();
}

export async function detectPhone(model, video) {
  const predictions = await model.detect(video);
  return predictions.some(pred => pred.class === "cell phone" && pred.score > 0.5);
} 