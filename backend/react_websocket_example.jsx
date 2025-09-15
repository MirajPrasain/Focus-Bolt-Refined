import React, { useRef, useEffect, useState, useCallback } from 'react';

// Configuration for different environments
const config = {
  development: {
    wsUrl: 'ws://localhost:8000/ws/study'
  },
  production: {
    wsUrl: 'wss://your-ngrok-url.ngrok-free.app/ws/study'
  }
};

// Use development by default, or set NODE_ENV=production for production
const environment = process.env.NODE_ENV || 'development';
const wsUrl = config[environment].wsUrl;

const StudySession = ({ duration = 30 }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);
  
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [focusScore, setFocusScore] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    try {
      // Use localhost for local testing instead of ngrok
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        setErrorCount(0);
        
        // Send session duration
        ws.send(JSON.stringify({ duration }));
      };
      
      ws.onmessage = (event) => {
        const data = event.data;
        
        // Handle error messages from server
        if (data.startsWith('error:')) {
          console.warn('Server error:', data);
          setConnectionStatus('error');
          return;
        }
        
        // Handle focus score
        const score = parseInt(data);
        if (!isNaN(score) && score >= 0 && score <= 100) {
          setFocusScore(score);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
        setErrorCount(prev => prev + 1);
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setConnectionStatus('disconnected');
        stopStreaming();
      };
      
      wsRef.current = ws;
      
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setConnectionStatus('error');
    }
  }, [duration]);

  // Frame validation and processing
  const validateCanvas = useCallback((canvas) => {
    if (!canvas) return false;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return false;
    
    // Check if canvas has content
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Check if canvas is not completely transparent/black
    let hasContent = false;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0 || data[i + 3] > 0) {
        hasContent = true;
        break;
      }
    }
    
    if (!hasContent) {
      console.warn('Canvas has no visible content');
      return false;
    }
    
    // Check minimum dimensions
    if (canvas.width < 100 || canvas.height < 100) {
      console.warn('Canvas dimensions too small:', canvas.width, 'x', canvas.height);
      return false;
    }
    
    return true;
  }, []);

  const captureAndSendFrame = useCallback(() => {
    if (!isConnected || !wsRef.current || !videoRef.current || !canvasRef.current) {
      return;
    }
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Validate video stream
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
      console.warn('Video not ready, skipping frame');
      return;
    }
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    } catch (error) {
      console.warn('Failed to draw video to canvas:', error);
      return;
    }
    
    // Validate canvas content
    if (!validateCanvas(canvas)) {
      return;
    }
    
    // Convert to base64
    let imageData;
    try {
      imageData = canvas.toDataURL('image/jpeg', 0.8);
    } catch (error) {
      console.warn('Failed to convert canvas to data URL:', error);
      return;
    }
    
    // Validate image data
    if (!imageData || imageData.length < 1000) {
      console.warn('Image data too small or invalid');
      return;
    }
    
    // Send frame to server
    try {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(imageData);
      }
    } catch (error) {
      console.error('Failed to send frame:', error);
      setErrorCount(prev => prev + 1);
    }
  }, [isConnected, validateCanvas]);

  // Start video stream
  const startStreaming = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 10 }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
        
        // Start frame capture loop
        const captureLoop = () => {
          captureAndSendFrame();
          animationFrameRef.current = requestAnimationFrame(captureLoop);
        };
        captureLoop();
      }
    } catch (error) {
      console.error('Failed to start video stream:', error);
      setConnectionStatus('error');
    }
  }, [captureAndSendFrame]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    setIsStreaming(false);
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // Start session
  const startSession = useCallback(() => {
    connectWebSocket();
  }, [connectWebSocket]);

  // Stop session
  const stopSession = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    stopStreaming();
    setIsConnected(false);
    setConnectionStatus('disconnected');
    setFocusScore(0);
  }, [stopStreaming]);

  // Auto-start streaming when connected
  useEffect(() => {
    if (isConnected && !isStreaming) {
      startStreaming();
    }
  }, [isConnected, isStreaming, startStreaming]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopSession();
    };
  }, [stopSession]);

  // Auto-reconnect on errors (with exponential backoff)
  useEffect(() => {
    if (errorCount > 0 && errorCount <= 3) {
      const timeout = setTimeout(() => {
        console.log(`Attempting to reconnect (attempt ${errorCount})`);
        connectWebSocket();
      }, Math.pow(2, errorCount) * 1000); // 2s, 4s, 8s backoff
      
      return () => clearTimeout(timeout);
    }
  }, [errorCount, connectWebSocket]);

  return (
    <div className="study-session">
      <div className="status-bar">
        <span className={`status ${connectionStatus}`}>
          {connectionStatus.toUpperCase()}
        </span>
        <span className="focus-score">Focus: {focusScore}%</span>
        {errorCount > 0 && <span className="error-count">Errors: {errorCount}</span>}
      </div>
      
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ display: isStreaming ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
      </div>
      
      <div className="controls">
        {!isConnected ? (
          <button onClick={startSession} className="start-btn">
            Start Study Session
          </button>
        ) : (
          <button onClick={stopSession} className="stop-btn">
            Stop Session
          </button>
        )}
      </div>
      
      <div className="session-info">
        <p>Duration: {duration} minutes</p>
        <p>Status: {isStreaming ? 'Streaming' : 'Not streaming'}</p>
      </div>
    </div>
  );
};

export default StudySession; 