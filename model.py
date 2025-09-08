import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np

# 1. Load the pre-trained MobileNetV2 model
# The model will be downloaded automatically the first time you run this.
# `weights='imagenet'` specifies loading weights pre-trained on the ImageNet dataset.
model = MobileNetV2(weights='imagenet')

# 2. Load and preprocess the image
# Replace 'my_image.jpg' with the path to your image file.
img_path = "Stop-Sign.v1i.tensorflow/valid/00014_00001_00025_png_jpg.rf.9b82bfc941ffc9127ad9c219516e4e27.jpg"
# MobileNetV2 expects images of size 224x224 pixels.
img = image.load_img(img_path, target_size=(224, 224)) 

# Convert the image to a NumPy array
img_array = image.img_to_array(img)

# Expand dimensions to create a "batch" of 1 image
# The model expects input of shape (batch_size, height, width, channels)
img_batch = np.expand_dims(img_array, axis=0)

# Preprocess the image for the MobileNetV2 model
# This scales pixel values to the range expected by the model.
img_preprocessed = preprocess_input(img_batch)

# 3. Make a prediction
prediction = model.predict(img_preprocessed)

# 4. Decode and print the prediction
# `decode_predictions` converts the raw prediction probabilities into human-readable labels.
# `top=3` means we want the top 3 most likely classifications.
decoded_prediction = decode_predictions(prediction, top=3)[0]

print("Prediction:")
for i, (imagenet_id, label, score) in enumerate(decoded_prediction):
    print(f"{i+1}: {label} ({score:.2%})")