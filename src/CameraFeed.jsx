import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import './CameraFeed.css';

function CameraFeed() {
  const videoRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState(null);
  const [doorState, setDoorState] = useState('closed');
  const [modelOutput, setModelOutput] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const streamRef = useRef(null);
  const modelRef = useRef(null);
  const predictionLoopRef = useRef(null);

  useEffect(() => {
    // Load the model
    const loadModel = async () => {
      try {
        // Optional but recommended
        await tf.setBackend('webgl');
        await tf.ready();

        const modelURL = `${window.location.origin}/model/model.json`;

        modelRef.current = await tf.loadLayersModel(modelURL);
        console.log('Loading model from:', modelURL);
        setModelLoaded(true);
      } catch (err) {
        console.error('Error loading model:', err);
        setError('Failed to load model');
      }
    };

    loadModel();

    return () => {
      // Cleanup: stop the stream when component unmounts
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (predictionLoopRef.current) {
        cancelAnimationFrame(predictionLoopRef.current);
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsActive(true);

        // Start prediction loop if model is loaded
        if (modelRef.current && modelLoaded) {
          startPredictionLoop();
        }
      }
    } catch (err) {
      setError(`Camera error: ${err.message}`);
      setIsActive(false);
    }
  };
  
const startPredictionLoop = () => {
  let lastPredictionTime = 0;
  const predictionDelay = 100;

  const predict = async () => {
    const now = Date.now();

    if (
      videoRef.current &&
      modelRef.current &&
      now - lastPredictionTime > predictionDelay
    ) {
      try {
        if (videoRef.current.readyState === videoRef.current.HAVE_ENOUGH_DATA) {
          // 1. Capture frame → tensor
          const inputTensor = tf.browser
            .fromPixels(videoRef.current)
            .resizeNearestNeighbor([224, 224]) // TM default
            .toFloat()
            .div(255.0)
            .expandDims(0); // [1, 224, 224, 3]

          // 2. Run inference
          const predictionTensor = modelRef.current.predict(inputTensor);

          // 3. Read results
          const probabilities = await predictionTensor.data();

          // 4. Cleanup
          inputTensor.dispose();
          predictionTensor.dispose();

          // 5. Find max probability
          let maxProb = 0;
          let maxIndex = 0;

          probabilities.forEach((prob, index) => {
            if (prob > maxProb) {
              maxProb = prob;
              maxIndex = index;
            }
          });

          // Define your labels manually (order matters!)
          const labels = ['closed', 'open'];

          const outputData = {
            output: maxIndex === 0 ? 0 : 1,
            confidence: (maxProb * 100).toFixed(1),
            class: labels[maxIndex] ?? 'Unknown'
          };

          console.log('Model Prediction:', outputData);

          setModelOutput(outputData);
          setDoorState(outputData.output === 0 ? 'open' : 'closed');

          lastPredictionTime = now;
        }
      } catch (err) {
        console.error('Prediction error:', err);
      }
    }

    predictionLoopRef.current = requestAnimationFrame(predict);
  };

  predict();
};

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setIsActive(false);
    }
    if (predictionLoopRef.current) {
      cancelAnimationFrame(predictionLoopRef.current);
    }
  };

  return (
    <div className="camera-container">
      <h1>Camera Feed</h1>
      
      {modelOutput && (
        <div className="model-output">
          <p className="model-text">
            Model Output: <span className="output-value">{modelOutput.output}</span>
          </p>
          <p className="model-text">
            Detection: <span className="output-value">{modelOutput.class}</span>
          </p>
          <p className="model-text">
            Confidence: <span className="output-value">{modelOutput.confidence}%</span>
          </p>
        </div>
      )}
      
      <div className="door-display">
        <img 
          src={doorState === 'closed' ? `${window.location.origin}/closed.jpg` : `${window.location.origin}/open.jpg`} 
          alt={`${doorState} door`} 
          className="door-image" 
        />
        {modelOutput && (
          <span className="door-prediction-emoji">
            {modelOutput.output === 1 ? '👊' : '🖐️'}
          </span>
        )}
      </div>
      
      <div className="video-wrapper">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="video-feed"
        />
      </div>

      {error && <p className="error-message">{error}</p>}

      <div className="button-group">
        <button
          onClick={startCamera}
          disabled={isActive}
          className="btn btn-start"
        >
          Start Camera
        </button>
        <button
          onClick={stopCamera}
          disabled={!isActive}
          className="btn btn-stop"
        >
          Stop Camera
        </button>
      </div>

      <p className="status">
        {isActive ? '🟢 Camera is running' : '⚪ Camera is off'}
      </p>
      {modelLoaded && <p className="status model-status">✓ Model loaded</p>}
    </div>
  );
}

export default CameraFeed;
