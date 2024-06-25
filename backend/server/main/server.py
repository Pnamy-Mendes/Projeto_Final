import os
import sys
import yaml
import socket
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from utils.helpers import find_available_port, load_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from server.main.predict_mood import predict_mood

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Load configuration
config = load_config('config.yaml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_mood', methods=['POST'])
def predict_mood_endpoint():
    return predict_mood(request)

@app.route('/history', methods=['GET'])
def history():
    try:
        history_path = os.path.join('static/results', 'history.json')
        history_data = []
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
        return jsonify({'history': history_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    return jsonify({'port': config['server']['port']})

if __name__ == '__main__':
    if not os.path.exists('static/results'):
        os.makedirs('static/results')

    available_port = find_available_port(config['server']['port'])
    config['server']['port'] = available_port

    with open('config.yaml', 'w') as file:
        yaml.safe_dump(config, file)

    app.run(host=config['server']['host'], port=available_port, debug=True, ssl_context=('cert.pem', 'key.pem'))
