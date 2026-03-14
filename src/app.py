from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "templates")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "pneumonia_model.h5")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load the trained model
model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    img = Image.open(file).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        result = "Pneumonia Detected"
    else:
        result = "Normal"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)