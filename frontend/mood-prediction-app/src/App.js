import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Chart from 'chart.js/auto';

const backendPort = process.env.REACT_APP_BACKEND_PORT || 5001;

const App = () => {
    const [mood, setMood] = useState('Unknown');
    const [confidence, setConfidence] = useState(0);
    const [error, setError] = useState('');
    const [history, setHistory] = useState([]);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const chartRef = useRef(null);
    const [isPredicting, setIsPredicting] = useState(false);
    const [currentStream, setCurrentStream] = useState(null);
    const [devices, setDevices] = useState([]);
    const [selectedDeviceId, setSelectedDeviceId] = useState('');

    useEffect(() => {
        loadHistory();
        getDevices();
        initializeChart();
    }, []);

    useEffect(() => {
        if (selectedDeviceId) {
            startStream();
        }
    }, [selectedDeviceId]);

    const loadHistory = async () => {
        try {
            const response = await axios.get(`https://127.0.0.1:${backendPort}/history`);
            setHistory(response.data.history);
        } catch (error) {
            setError(`Error loading history: ${error.message}`);
            console.error(error);
        }
    };

    const startPrediction = async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');

        if (video && context) {
            setIsPredicting(true);

            const captureFrame = () => {
                if (!isPredicting) return;

                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg');

                axios.post(`https://127.0.0.1:${backendPort}/predict_mood`, { image: imageData })
                    .then(response => {
                        if (response.data.error) {
                            setError(response.data.error);
                            return;
                        }

                        setMood(response.data.mood);
                        setConfidence(response.data.confidence);
                        updateGraph(response.data.confidence);
                        updateTimeline(response.data.image_path, response.data.mood, response.data.confidence);
                    })
                    .catch(error => {
                        setError(`Prediction error: ${error.message}`);
                        console.error(error);
                    });

                requestAnimationFrame(captureFrame);
            };

            captureFrame();
        }
    };

    const stopPrediction = () => {
        setIsPredicting(false);
    };

    const initializeChart = () => {
        const ctx = document.getElementById('mood-graph').getContext('2d');
        chartRef.current = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Confidence',
                    data: [],
                    backgroundColor: 'rgba(0, 123, 255, 0.5)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { display: true },
                    y: { beginAtZero: true, max: 100 }
                }
            }
        });
    };

    const updateGraph = (newConfidence) => {
        const chart = chartRef.current;
        if (chart) {
            const labels = chart.data.labels;
            const data = chart.data.datasets[0].data;
            labels.push('');
            data.push(newConfidence);

            if (labels.length > 20) {
                labels.shift();
                data.shift();
            }

            chart.update();
        }
    };

    const updateTimeline = (imagePath, mood, confidence) => {
        setHistory(prevHistory => [
            ...prevHistory,
            { image_path: imagePath, mood, confidence }
        ]);
    };

    const startStream = async () => {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        const constraints = {
            video: {
                deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined
            }
        };
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            setCurrentStream(stream);
            videoRef.current.srcObject = stream;
        } catch (error) {
            console.error('Error starting video stream:', error);
            setError('Error starting video stream');
        }
    };

    const getDevices = async () => {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            setDevices(videoDevices);
            if (videoDevices.length > 0) {
                setSelectedDeviceId(videoDevices[0].deviceId);
            }
        } catch (error) {
            console.error('Error getting devices:', error);
        }
    };

    return (
        <div>
            <h1>Webcam Mood Prediction</h1>
            <div>
                <div>
                    <label htmlFor="videoSource">Choose a camera:</label>
                    <select id="videoSource" value={selectedDeviceId} onChange={(e) => setSelectedDeviceId(e.target.value)}>
                        {devices.map((device, index) => (
                            <option key={device.deviceId} value={device.deviceId}>
                                {device.label || `Camera ${index + 1}`}
                            </option>
                        ))}
                    </select>
                </div>
                <video ref={videoRef} autoPlay playsInline width="640" height="480"></video>
                <button onClick={isPredicting ? stopPrediction : startPrediction}>
                    {isPredicting ? 'Stop Prediction' : 'Start Prediction'}
                </button>
                <div>Mood: {mood}</div>
                <div>Confidence: {confidence}%</div>
                <div style={{ color: 'red' }}>{error}</div>
                <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }}></canvas>
                <div>
                    <canvas id="mood-graph"></canvas>
                </div>
            </div>
            <div>
                <h2>Prediction History</h2>
                <div className="carousel">
                    {history.map((item, index) => (
                        <div key={index}>
                            <img src={`https://127.0.0.1:${backendPort}${item.image_path}`} alt={`Mood: ${item.mood}`} />
                            <div>Mood: {item.mood}</div>
                            <div>Confidence: {item.confidence}%</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default App;
