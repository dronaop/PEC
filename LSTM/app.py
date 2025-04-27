from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the model and scaler
model = load_model('New_model.keras')
scaler = joblib.load('scaler.pkl') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("input")

       
        if not data or len(data) != 5:
            return jsonify({"error": "Input must be a list of 5 timesteps, each with 6 values"}), 400
        
        user_input = np.array(data)
        if user_input.shape != (5, 6):
            return jsonify({"error": "Each timestep must have exactly 6 features"}), 400
        
       
        scaled_input = scaler.transform(user_input)
        scaled_input = scaled_input.reshape(1, 5, 6)

        
        prediction = model.predict(scaled_input)

       
        pad = np.zeros((prediction.shape[0], scaler.n_features_in_ - prediction.shape[1]))
        padded_pred = np.concatenate([prediction, pad], axis=1)
        prediction_unscaled = scaler.inverse_transform(padded_pred)[:, :6]  # only the first 6

        return jsonify({
            "Predicted Output": prediction_unscaled[0].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(debug=True)