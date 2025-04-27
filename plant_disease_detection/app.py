import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import tensorflow as tf
import numpy as np
from PIL import Image
import io

import tensorflow as tf
print(tf.__version__)
print(tf.keras.models)

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('trained_model.keras', compile=False)

# List of class names
class_name = [
    'Apple Apple scab',
    'Apple Black rot',
    'Apple Cedar apple rust',
    'Apple healthy',
    'Blueberry healthy',
    'Cherry (including sour) Powdery mildew',
    'Cherry (including sour) healthy',
    'Corn (maize) Cercospora leaf spot Gray leaf spot',
    'Corn (maize) Common rust',
    'Corn (maize) Northern Leaf Blight',
    'Corn (maize) healthy',
    'Grape Black rot',
    'Grape Esca (Black Measles)',
    'Grape Leaf blight (Isariopsis Leaf Spot)',
    'Grapehealthy',
    'Orange Haunglongbing (Citrus greening)',
    'Peach Bacterial spot',
    'Peach healthy',
    'Pepper, bell Bacterial spot',
    'Pepper, bell healthy',
    'Potato Early blight',
    'Potato Late blight',
    'Potato healthy',
    'Raspberry healthy',
    'Soybean healthy',
    'Squash Powdery mildew',
    'Strawberry Leaf scorch',
    'Strawberry healthy',
    'Tomato Bacterial spot',
    'Tomato Early blight',
    'Tomato Late blight',
    'Tomato Leaf Mold',
    'Tomato Septoria leaf spot',
    'Tomato Spider mites Two-spotted spider mite',
    'Tomato Target Spot',
    'Tomato Tomato Yellow Leaf Curl Virus',
    'Tomato Tomato mosaic virus',
    'Tomato healthy'
]

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((128, 128))  # Resize image to match model input size
        image_array = np.array(image) # Normalize pixel values
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error in processing image: {str(e)}")

@app.route('/')
def home():
    return render_template('model_form.html')  # Render the form

@app.route('/predict', methods=['POST'])
def predict():
    try:

        # Get the image file from the POST request
        file = request.files['image']
        image_bytes = file.read()
        
        # Preprocess the image
        input_data = preprocess_image(image_bytes)
        
        # Get predictions from the model
        predictions = model.predict(input_data)
        
        # Get the predicted class index and confidence
        predicted_class = np.argmax(predictions, axis=-1)[0]
        confidence = float(np.max(predictions))
        
        # Map the class index to the class name
        predicted_label = class_name[predicted_class]

        # Return the response with the prediction
        return jsonify({
            'predicted_class': int(predicted_class),
            'class_label': predicted_label,
            'confidence': confidence
        })
    
    except Exception as e:
        # Return error message if an exception occurs
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Start the Flask app