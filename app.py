from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
model = None

def load_model():
    global model
    print("Loading model...")
    model = tf.keras.models.load_model("pneumonia_model.", compile=False)
    print("Model loaded! ✅")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        result = float(model.predict(img_array, verbose=0)[0][0])
        if result > 0.5:
            label = "PNEUMONIA"
            confidence = round(result * 100, 1)
            risk = "HIGH" if confidence > 80 else "MODERATE"
        else:
            label = "NORMAL"
            confidence = round((1 - result) * 100, 1)
            risk = "LOW"
        return jsonify({'label': label, 'confidence': confidence, 'risk': risk, 'score': round(result, 3)})
    except Exception as e:
        print("ERROR:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    print("\n Open browser: http://localhost:5000\n")
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

