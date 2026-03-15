import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model("pneumonia_model.h5")

# Put your image path here
IMG_PATH = "test_xray.jpg.jpeg"  # <-- change this to your image file name

# Prepare image
img = image.load_img(IMG_PATH, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
result = model.predict(img_array)[0][0]

if result > 0.5:
    print(f"Result: PNEUMONIA detected  (confidence: {result*100:.1f}%)")
else:
    print(f"Result: NORMAL  (confidence: {(1-result)*100:.1f}%)")
