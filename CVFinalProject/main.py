import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from flask import Flask, request, render_template
from PIL import Image
from scipy.io import loadmat
import cv2
import base64
from io import BytesIO

# Initialize Flask application
app = Flask(__name__)

# Path to the dataset (adjust the paths according to your dataset location)
DATASET_PATH = "C:/Users/saiak/Downloads/ShanghaiTech_Crowd_Counting_Dataset"
PART_A_TRAIN = os.path.join(DATASET_PATH, 'part_A_final', 'train_data')
PART_B_TRAIN = os.path.join(DATASET_PATH, 'part_B_final', 'train_data')
PART_A_TEST = os.path.join(DATASET_PATH, 'part_A_final', 'test_data')
PART_B_TEST = os.path.join(DATASET_PATH, 'part_B_final', 'test_data')


# Load dataset images and annotations
def load_dataset(part_folder, data_type='train'):
    images = []
    labels = []
    image_files = sorted([f for f in os.listdir(part_folder) if f.endswith('.jpg')])
    for img_file in image_files:
        img_path = os.path.join(part_folder, img_file)
        mat_file = img_file.replace('.jpg', '.mat')
        mat_path = os.path.join(part_folder, mat_file)

        # Load the image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img) / 255.0  # Normalize the image

        # Load the annotations from the .mat file
        mat_data = loadmat(mat_path)
        image_info = mat_data.get("image_info", None)
        if image_info is not None:
            head_coords = image_info[0][0]['location']
            head_coords = np.array(head_coords)
            density_map = create_density_map(head_coords)
            images.append(img)
            labels.append(density_map)

    return np.array(images), np.array(labels)


# Create density map from head locations
def create_density_map(head_coords, img_size=(224, 224), sigma=2):
    density_map = np.zeros(img_size)
    for coord in head_coords:
        x, y = int(coord[0]), int(coord[1])
        if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
            density_map[x, y] = 1  # Mark the location of each head
    # Apply Gaussian filter to create a density map
    density_map = cv2.GaussianBlur(density_map, (sigma, sigma), 0)
    return density_map


# Build the CNN model for crowd density estimation
def build_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(224 * 224, activation='sigmoid'),  # Output: a flattened density map
        layers.Reshape((224, 224))  # Reshape to match the input image size
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Train the model
def train_model(model, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=32):
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs,
              batch_size=batch_size)


# Convert an image to base64 for display on the web
def encode_image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# Prediction route for Flask
@app.route('/predict', methods=['POST'])
def predict():
    file_image = request.files['image']

    # Preprocess the image for prediction
    image = Image.open(file_image).convert('RGB')
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Load the model (assuming the model is already trained and saved)
    model = tf.keras.models.load_model('crowd_density_model.h5')

    # Predict density map
    pred_density_map = model.predict(image_array)[0]

    # Convert the density map to an image
    pred_density_map_img = (pred_density_map * 255).astype(np.uint8)
    pred_density_map_img = Image.fromarray(pred_density_map_img)

    # Convert images to base64 for web display
    uploaded_img_str = encode_image_to_base64(image_resized)
    pred_img_str = encode_image_to_base64(pred_density_map_img)

    return render_template(
        'result.html',
        uploaded_img=uploaded_img_str,
        pred_img=pred_img_str
    )


# Home route
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
