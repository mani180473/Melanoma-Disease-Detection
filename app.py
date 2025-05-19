import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and label encoder
model = load_model('melanoma_model.h5')
label_encoder = LabelEncoder()
label_encoder.fit(['benign', 'malignant'])  # Make sure it matches training labels

# Prediction function
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    class_name = label_encoder.inverse_transform([predicted_class])[0]
    return class_name, confidence

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    class_name, confidence = predict_image(filepath)
    return render_template('index.html', filename=file.filename, prediction=class_name, confidence=confidence)

@app.route('/display/<filename>')
def display_image(filename):
    return f'/static/uploads/{filename}'

if __name__ == '__main__':
    app.run(debug=True)
