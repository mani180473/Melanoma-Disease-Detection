import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session

# Clear any previous sessions
clear_session()

# Define directories (update these paths as needed)
base_dir = r'C:\Users\vinnu\Desktop\melanoma\train'  # Training data
test_dir = r'C:\Users\vinnu\Desktop\melanoma\test'    # Test data

# Function to load images and labels from a directory
def load_data(directory):
    X = []
    y = []
    class_labels = os.listdir(directory)  # Get class subdirectories
    for label in class_labels:
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                try:
                    # Load and preprocess the image
                    img = load_img(image_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    X.append(img_array)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    return np.array(X, dtype='float32') / 255.0, np.array(y)

# Load training and test data
X_train, y_train = load_data(base_dir)
X_test, y_test = load_data(test_dir)

print("Data loaded successfully!")
print(f"Training images: {len(X_train)}, Training labels: {len(y_train)}")
print(f"Test images: {len(X_test)}, Test labels: {len(y_test)}")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Split the data for validation
X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42
)

# One-hot encode labels
y_train_categorical = to_categorical(y_train_encoded)
y_val_categorical = to_categorical(y_val_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
out = Dense(len(np.unique(y_train_encoded)), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=out)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train_categorical,
    validation_data=(X_val, y_val_categorical),
    batch_size=32,
    epochs=20
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# Predictions
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Classification report
print(classification_report(y_test_encoded, y_pred_labels, target_names=label_encoder.classes_))

# Confusion matrix
conf_mat = confusion_matrix(y_test_encoded, y_pred_labels)
plt.figure(figsize=(8, 8))
plt.imshow(conf_mat, cmap='Reds')
plt.title('ResNet50 Confusion Matrix')
plt.colorbar()
plt.show()

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()

# Save the model in .h5 format
model.save('melanoma_model.h5')
print("Model saved as melanoma_model.h5")
