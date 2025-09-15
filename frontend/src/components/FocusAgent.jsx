import React, { useRef, useEffect, useState } from "react";
import { useWebcam } from "../hooks/useWebcam";
import { loadFaceMeshModel } from "../detectors/faceMeshDetector";
import { loadPhoneModel } from "../detectors/phoneDetector";
import { processFrame } from "../agents/FocusAgentController";

export default function FocusAgent() {
  const videoRef = useRef(null);
  const [status, setStatus] = useState({});
  useWebcam(videoRef);

  useEffect(() => {
    let faceModel, phoneModel, running = true;
    async function init() {
      faceModel = await loadFaceMeshModel();
      phoneModel = await loadPhoneModel();
      async function loop() {
        if (!running) return;
        await processFrame({
          faceModel,
          phoneModel,
          video: videoRef.current,
          onResults: setStatus
        });
        requestAnimationFrame(loop);
      }
      loop();
    }
    init();
    return () => { running = false; };
  }, []);

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline width={640} height={480} />
      <div>
        <p>Blink: {status.blink ? "Yes" : "No"}</p>
        <p>Looking: {status.looking ? "Yes" : "No"}</p>
        <p>Head Tilt: {status.tilt?.toFixed(2)}°</p>
        <p>Phone Detected: {status.phoneDetected ? "Yes" : "No"}</p>
        <p>Multiple Faces: {status.multiFace ? "Yes" : "No"}</p>
      </div>
    </div>
  );
} 