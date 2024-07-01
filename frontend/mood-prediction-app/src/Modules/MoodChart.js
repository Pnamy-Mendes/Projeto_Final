import React from 'react';
import { Line } from 'react-chartjs-2';

function MoodChart({ history }) {
    const data = {
        labels: history.map((_, index) => index),
        datasets: [{
            label: 'Confidence',
            data: history.map(item => item.confidence),
            backgroundColor: 'rgba(0, 255, 255, 0.2)',
            borderColor: 'rgba(0, 255, 255, 1)',
            borderWidth: 1
        }]
    };

    const options = {
        scales: {
            x: { display: false },
            y: { beginAtZero: true, max: 100 }
        }
    };

    return <Line data={data} options={options} />;
}

export default MoodChart;
