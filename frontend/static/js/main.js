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

    const moodLabelMap = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
    };

    const moodColors = {
        0: 'rgba(255, 0, 0, 0.2)',  // angry
        1: 'rgba(128, 0, 128, 0.2)',  // disgust
        2: 'rgba(0, 0, 255, 0.2)',  // fear
        3: 'rgba(255, 255, 0, 0.2)',  // happy
        4: 'rgba(0, 0, 139, 0.2)',  // sad
        5: 'rgba(255, 165, 0, 0.2)',  // surprise
        6: 'rgba(128, 128, 128, 0.2)'  // neutral
    };

    const raceLabelMap = {
        0: "White",
        1: "Black",
        2: "Asian",
        3: "Indian",
        4: "Other"
    };

    const raceColors = {
        0: 'rgba(255, 99, 132, 0.2)',  // White
        1: 'rgba(54, 162, 235, 0.2)',  // Black
        2: 'rgba(75, 192, 192, 0.2)',  // Asian
        3: 'rgba(153, 102, 255, 0.2)',  // Indian
        4: 'rgba(255, 159, 64, 0.2)'  // Other
    };

    const moodChartContext = document.getElementById('mood-graph').getContext('2d');
    const moodChart = new Chart(moodChartContext, {
        type: 'line',
        data: {
            labels: [],
            datasets: Object.keys(moodLabelMap).map(key => ({
                label: moodLabelMap[key],
                data: [],
                backgroundColor: moodColors[key],
                borderColor: moodColors[key].replace('0.2', '1'),
                borderWidth: 1
            }))
        },
        options: {
            responsive: false,
            scales: {
                x: { display: false },
                y: {
                    beginAtZero: true, max: 100,
                    ticks: {
                        stepSize: 25,
                    }
                }
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
        type: 'bar',
        data: {
            labels: ['Male', 'Female'],
            datasets: [{
                label: 'Gender Confidence',
                data: [0, 0],
                backgroundColor: ['rgba(54, 162, 235, 0.2)', 'rgba(255, 99, 132, 0.2)'],
                borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: false,
            scales: {
                x: { display: true },
                y: { beginAtZero: true, max: 100 }
            }
        }
    });

    const raceChartContext = document.getElementById('race-graph').getContext('2d');
    const raceChart = new Chart(raceChartContext, {
        type: 'line',
        data: {
            labels: [],
            datasets: Object.keys(raceLabelMap).map(key => ({
                label: raceLabelMap[key],
                data: [],
                backgroundColor: raceColors[key],
                borderColor: raceColors[key].replace('0.2', '1'),
                borderWidth: 1
            }))
        },
        options: {
            responsive: false,
            scales: {
                x: { display: false },
                y: {
                    beginAtZero: true, max: 100,
                    ticks: {
                        stepSize: 25,
                    }
                }
            }
        }
    });

    function updateGraph(chart, datasetIndex, value) {
        if (!chart.data.datasets[datasetIndex]) {
            console.error(`Dataset index ${datasetIndex} is not defined.`);
            return;
        }

        const labels = chart.data.labels;
        const data = chart.data.datasets[datasetIndex].data;
        const maxDataPoints = 20;

        labels.push('');
        data.push(value);

        if (labels.length > maxDataPoints) {
            labels.shift();
            chart.data.datasets.forEach(dataset => dataset.data.shift());
        }

        chart.update();
    }

    function updateTimeline(imagePath, mood, mood_confidence) {
        const carouselSlides = document.querySelector('.carousel__slides');
        const carouselThumbnails = document.querySelector('.carousel__thumbnails');
        const slideIndex = carouselSlides.children.length + 1;
    
        const slide = document.createElement('li');
        slide.classList.add('carousel__slide');
        slide.innerHTML = `
            <figure>
                <div>
                    <img src="${imagePath}" alt="">
                </div>
                <figcaption>
                    Mood: ${mood} (${mood_confidence}%)
                </figcaption>
            </figure>
        `;
    
        const thumbnail = document.createElement('li');
        thumbnail.innerHTML = `
            <label for="slide-${slideIndex}">
                <img src="${imagePath}" alt="">
            </label>
        `;
    
        carouselSlides.appendChild(slide);
        carouselThumbnails.appendChild(thumbnail);
    
        if (slideIndex > 6) {
            const newInput = document.createElement('input');
            newInput.type = 'radio';
            newInput.name = 'slides';
            newInput.id = `slide-${slideIndex}`;
            document.querySelector('.carousel').appendChild(newInput);
        }
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
                const { mood, mood_confidence, age, gender, race, image_path } = result;
                moodDisplay.textContent = `Mood: ${mood} (${mood_confidence}%)`;
                moodDisplay.style.color = mood_confidence > 70 ? 'green' : mood_confidence > 40 ? 'yellow' : 'red';
                ageDisplay.textContent = `Age: ${age}`;
                genderDisplay.textContent = `Gender: ${gender}`;
                raceDisplay.textContent = `Race: ${race}`;
                updateGraph(moodChart, Object.keys(moodLabelMap).find(key => moodLabelMap[key] === mood), mood_confidence);
                updateGraph(ageChart, 0, age);
                genderChart.data.datasets[0].data = [gender === 'Male' ? mood_confidence : 0, gender === 'Female' ? mood_confidence : 0]; // Update gender confidence graph
                genderChart.update();
                updateGraph(raceChart, Object.keys(raceLabelMap).find(key => raceLabelMap[key] === race), mood_confidence);  // Update race confidence graph
                updateTimeline(image_path, mood, mood_confidence);
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
                updateTimeline(item.image_path, item.mood, item.mood_confidence);
            });
            } catch (error) {
            console.error(`Error loading history: ${error.message}`);
        }
    }

    loadHistory();
});
