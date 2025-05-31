from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

import tensorflow as tf
import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
app = Flask(__name__)

from flask_cors import CORS
CORS(app)


model = load_model('best_model.keras', compile=False)
scaler = joblib.load('scaler1.pkl') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("input")

       
        if not data or len(data) != 5:
            return jsonify({"error": "Input must be a list of 5 timesteps, each with 6 values"}), 400
        
        
        columns = [
            'Ambient_Temperature', 'Soil_Temperature', 'Humidity', 
            'Light_Intensity', 'Chlorophyll_Content', 'Electrochemical_Signal'
        ]
        
        
        user_input = pd.DataFrame(data, columns=columns)
        
        if user_input.shape != (5, 6):
            return jsonify({"error": "Each timestep must have exactly 6 features"}), 400
        
       
        scaled_input = scaler.transform(user_input)  
        scaled_input = scaled_input.reshape(1, 5, 6)

        
        prediction = model.predict(scaled_input)

       
        pad = np.zeros((prediction.shape[0], scaler.n_features_in_ - prediction.shape[1]))
        padded_pred = np.concatenate([prediction, pad], axis=1)
        prediction_unscaled = scaler.inverse_transform(padded_pred)[:, :6]  

        return jsonify({
            "Predicted Output": prediction_unscaled[0].tolist()
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(debug=True)
