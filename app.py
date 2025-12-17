from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract features
    features = np.array([[
        data['pregnancies'],
        data['glucose'],
        data['bloodPressure'],
        data['skinThickness'],
        data['insulin'],
        data['bmi'],
        data['diabetesPedigree'],
        data['age']
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = int(probability >= 0.45)  # Your optimal threshold
    
    return jsonify({
        'probability': float(probability * 100),
        'prediction': prediction,
        'risk_level': 'High' if probability >= 0.6 else 'Moderate' if probability >= 0.3 else 'Low'
    })

if __name__ == '__main__':
    app.run(debug=True)