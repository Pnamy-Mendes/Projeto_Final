document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById('webcam');
    const cameraSelect = document.getElementById('camera-select');
    const toggleButton = document.getElementById('toggle-button');
    const moodDisplay = document.getElementById('mood');
    const ageDisplay = document.getElementById('age');
    const genderDisplay = document.getElementById('gender');
    const raceDisplay = document.getElementById('race');
    const errorDisplay = document.getElementById('error');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    let currentStream;
    let predictionInterval;
    let isPredicting = false;

    const chartContext = document.getElementById('mood-graph').getContext('2d');
    const moodChart = new Chart(chartContext, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Mood Confidence',
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

    const ageChartContext = document.getElementById('age-graph').getContext('2d');
    const ageChart = new Chart(ageChartContext, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Age Prediction',
                data: [],
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: false,
            scales: {
                x: { display: false },
                y: { beginAtZero: true }
            }
        }
    });

    const genderChartContext = document.getElementById('gender-graph').getContext('2d');
    const genderChart = new Chart(genderChartContext, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Gender Confidence',
                data: [],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
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

    const raceChartContext = document.getElementById('race-graph').getContext('2d');
    const raceChart = new Chart(raceChartContext, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Race Confidence',
                data: [],
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
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

    function updateGraph(chart, confidence) {
        const labels = chart.data.labels;
        const data = chart.data.datasets[0].data;
        const maxDataPoints = 20;

        labels.push('');
        data.push(confidence);

        if (labels.length > maxDataPoints) {
            labels.shift();
            data.shift();
        }

        chart.update();
    }

    async function getConnectedDevices(type) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices.filter(device => device.kind === type);
    }

    async function startStream() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        const videoConstraints = {};
        if (cameraSelect.value) {
            videoConstraints.deviceId = { exact: cameraSelect.value };
        }
        const constraints = {
            video: videoConstraints,
            audio: false
        };
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
    }

    async function updateCameraList() {
        const cameras = await getConnectedDevices('videoinput');
        cameraSelect.innerHTML = '';
        cameras.forEach(camera => {
            const option = document.createElement('option');
            option.value = camera.deviceId;
            option.text = camera.label || `Camera ${cameraSelect.length + 1}`;
            cameraSelect.appendChild(option);
        });
    }

    cameraSelect.addEventListener('change', startStream);
    toggleButton.addEventListener('click', async () => {
        if (isPredicting) {
            clearInterval(predictionInterval);
            isPredicting = false;
            toggleButton.textContent = 'Start Prediction';
            return;
        }

        const stream = video.srcObject;
        if (!stream) {
            errorDisplay.textContent = 'Error: No webcam stream available';
            return;
        }

        const track = stream.getVideoTracks()[0];
        const settings = track.getSettings();
        canvas.width = settings.width || 640;
        canvas.height = settings.height || 480;
        errorDisplay.textContent = '';
        isPredicting = true;
        toggleButton.textContent = 'Stop Prediction';

        predictionInterval = setInterval(async () => {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg');
            try {
                const response = await fetch('/predict_mood', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ image: imageData })
                });
                const result = await response.json();
                if (result.error) {
                    throw new Error(result.error);
                }
                const { mood, confidence_mood, age, gender, race, image_path } = result;
                moodDisplay.textContent = `Mood: ${mood} (${confidence_mood}%)`;
                moodDisplay.style.color = confidence_mood > 70 ? 'green' : confidence_mood > 40 ? 'yellow' : 'red';
                ageDisplay.textContent = `Age: ${age}`;
                genderDisplay.textContent = `Gender: ${gender}`;
                raceDisplay.textContent = `Race: ${race}`;
                updateGraph(moodChart, confidence_mood);
                updateGraph(ageChart, age);
                updateGraph(genderChart, confidence_mood); // Adjust based on your needs
                updateGraph(raceChart, confidence_mood);  // Adjust based on your needs
            } catch (error) {
                errorDisplay.textContent = `Error: ${error.message}`;
            }
        }, 1000);
    });

    updateCameraList().then(startStream);

    // Load history on page load
    async function loadHistory() {
        try {
            const response = await fetch('/history');
            const result = await response.json();
            if (result.error) {
                throw new Error(result.error);
            }
            result.history.forEach(item => {
                updateGraph(moodChart, item.confidence_mood);
                updateGraph(ageChart, item.age);
                updateGraph(genderChart, item.confidence_mood); // Adjust based on your needs
                updateGraph(raceChart, item.confidence_mood);  // Adjust based on your needs
            });
        } catch (error) {
            console.error(`Error loading history: ${error.message}`);
        }
    }

    loadHistory();
});
