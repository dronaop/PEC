import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import tensorflow as tf
import numpy as np
from PIL import Image
import io


print(tf.keras.models)

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model('trained_model.keras', compile=False)


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
        image = image.resize((128, 128))  
        image_array = np.array(image) 
        return np.expand_dims(image_array, axis=0) 
    except Exception as e:
        raise ValueError(f"Error in processing image: {str(e)}")

@app.route('/')
def home():
    return render_template('model_form.html')  

@app.route('/predict', methods=['POST'])
def predict():
    try:

        
        file = request.files['image']
        image_bytes = file.read()
        
        
        input_data = preprocess_image(image_bytes)
        
        
        predictions = model.predict(input_data)
        
        
        predicted_class = np.argmax(predictions, axis=-1)[0]
        confidence = float(np.max(predictions))
        
        predicted_label = class_name[predicted_class]

        return jsonify({
            'predicted_class': int(predicted_class),
            'class_label': predicted_label,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)  
