import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Chart from 'chart.js/auto';

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
    const [backendPort, setBackendPort] = useState(5001);

    useEffect(() => {
        fetchBackendPort();
    }, []);

    useEffect(() => {
        if (backendPort) {
            loadHistory();
        }
    }, [backendPort]);

    const fetchBackendPort = async () => {
        try {
            const response = await axios.get('http://127.0.0.1:5001/config');
            setBackendPort(response.data.port);
        } catch (error) {
            console.error('Error fetching backend port:', error);
        }
    };

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

    useEffect(() => {
        if (!chartRef.current) {
            const ctx = document.getElementById('mood-graph').getContext('2d');
            chartRef.current = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Confidence',
                        data: [],
                        backgroundColor: 'rgba(0, 255, 255, 0.2)',
                        borderColor: 'rgba(0, 255, 255, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: false,
                    scales: {
                        x: { display: false },
                        y: { beginAtZero: true, max: 100 }
                    }
                }
            });
        }
    }, []);

    const startStream = async () => {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        const constraints = { video: true };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        setCurrentStream(stream);
        videoRef.current.srcObject = stream;
    };

    useEffect(() => {
        startStream();
    }, []);

    return (
        <div>
            <h1>Webcam Mood Prediction</h1>
            <div>
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
