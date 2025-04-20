from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load ML model and preprocessing tools
model = joblib.load('fog_classification_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        temp = float(data['temp'])
        pm25 = float(data['pm25'])
        aqi = float(data['aqi'])
        pm10 = float(data['pm10'])
        humidity = float(data['humidity'])

        # Input in model's expected order
        model_input = np.array([[aqi, pm10, pm25, temp, humidity]])
        scaled_input = scaler.transform(model_input)
        prediction = model.predict(scaled_input)
        result = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
