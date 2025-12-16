from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load model from .keras format
model = load_model("nsfw_5_classes_final.h5")

# Modify based on your model's input
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # normalize if needed
    image = np.expand_dims(image, axis=0)  # shape: (1, 224, 224, 3)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    try:
        img = Image.open(file)
        input_data = preprocess_image(img)
        prediction = model.predict(input_data)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
