import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('melanoma_model.h5')

# Recreate the same label encoder used during training
label_encoder = LabelEncoder()
label_encoder.fit(['benign', 'malignant'])  # Replace with actual class labels if more

# Prediction function
def predict_image(image_path, model, label_encoder):
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        class_name = label_encoder.inverse_transform([predicted_class])[0]

        # Display image with prediction
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Predicted: {class_name} ({prediction[0][predicted_class] * 100:.2f}%)")
        plt.show()

    except Exception as e:
        print(f"Error processing image: {e}")

# Example usage
image_path = r'C:\Users\vinnu\Desktop\melanoma\test\benign\melanoma_9607.jpg'
predict_image(image_path, model, label_encoder)
