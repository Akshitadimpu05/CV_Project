import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, render_template
from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D, ReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)

# Paths to dataset
image_folder = "C:/Users/saiak/Downloads/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images/"
gt_folder = "C:/Users/saiak/Downloads/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth/"

# Function to load data
def load_data(image_folder, gt_folder):
    image_paths = sorted(os.listdir(image_folder))
    gt_paths = sorted(os.listdir(gt_folder))

    images = []
    ground_truths = []

    for img_name, gt_name in zip(image_paths, gt_paths):
        # Load image
        image = load_img(os.path.join(image_folder, img_name), target_size=(224, 224))
        image = img_to_array(image) / 255.0  # Normalize
        images.append(image)

        # Load ground truth density map
        gt_data = loadmat(os.path.join(gt_folder, gt_name))
        head_coords = gt_data['image_info'][0][0][0][0][0]
        density_map = generate_density_map(image, head_coords)
        ground_truths.append(density_map)

    return np.array(images), np.array(ground_truths)

# Function to generate density map
def generate_density_map(image, head_coords, target_size=(224, 224), sigma=4):
    # Get the original image dimensions from the numpy array
    original_height, original_width, _ = image.shape  # Image shape is (height, width, channels)

    # Resize the density map to match the target size
    density_map = np.zeros(target_size)

    # Scale the coordinates to the target size
    for coord in head_coords:
        x, y = coord[0], coord[1]

        # Scale the coordinates from original to target size
        x_scaled = int(np.round(x * target_size[0] / original_width))
        y_scaled = int(np.round(y * target_size[1] / original_height))

        # Ensure coordinates are within the bounds of the resized image
        x_scaled = min(max(x_scaled, 0), target_size[0] - 1)
        y_scaled = min(max(y_scaled, 0), target_size[1] - 1)

        # Place a 1 in the density map at the scaled coordinates
        density_map[y_scaled, x_scaled] = 1

    # Apply Gaussian filter to generate a smooth density map
    density_map = gaussian_filter(density_map, sigma=sigma)

    return density_map

# Check if the model already exists, if not, train and save it
if not os.path.exists('model/model.h5'):
    # Load data
    images, ground_truths = load_data(image_folder, gt_folder)

    # Define the model input
    input_layer = Input(shape=(224, 224, 3))
    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=input_layer)

    # Add custom layers for density map estimation
    x = vgg.output
    x = Conv2D(512, (3, 3), padding='same', dilation_rate=2, activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    # Upsample the feature map from (7, 7) to (224, 224)
    x = UpSampling2D(size=(32, 32))(x)  # Upscale 7x7 to 224x224

    # Final output layer (density map)
    output_layer = Conv2D(1, (1, 1), padding='same')(x)  # 1 channel for density map
    output_layer = ReLU()(output_layer)  # Apply ReLU to ensure positive values in density map

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(images, ground_truths, batch_size=4, epochs=10)

    # Save the trained model
    model.save('model/model.h5')
    print("Model trained and saved as model.h5")
else:
    # Load the pre-trained model
    model = load_model('model/model.h5')
    print("Model loaded from model.h5")

# Function to evaluate the model using MAE and RMSE
def evaluate_model(model, images, ground_truths):
    mae = 0
    mse = 0
    total_images = len(images)

    for i in range(total_images):
        # Get predicted density map
        image_input = np.expand_dims(images[i], axis=0)  # Add batch dimension
        predicted_density_map = model.predict(image_input)
        predicted_density_map = np.squeeze(predicted_density_map)  # Remove batch dimension

        # Get actual and predicted counts
        actual_count = np.sum(ground_truths[i])  # Sum of ground truth density map
        predicted_count = np.sum(predicted_density_map)  # Sum of predicted density map

        # Update MAE and MSE
        mae += abs(predicted_count - actual_count)
        mse += (predicted_count - actual_count) ** 2

    # Average over all images
    mae /= total_images
    mse = np.sqrt(mse / total_images)  # Root Mean Squared Error

    return mae, mse

# Evaluate the model
mae, rmse = evaluate_model(model, images, ground_truths)
print(f"MAE: {mae}, RMSE: {rmse}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files['image']
        mat_file = request.files['mat_file']

        img = Image.open(image)
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize image

        # Process .mat file if present
        if mat_file:
            mat_data = loadmat(mat_file)
            head_coords = mat_data['image_info'][0][0][0][0][0]
            gt_density_map = generate_density_map(img_array, head_coords)

        # Predict density map using the model
        image_input = np.expand_dims(img_array, axis=0)
        predicted_density_map = model.predict(image_input)
        predicted_density_map = np.squeeze(predicted_density_map)

        # Generate base64 images to send to frontend
        pred_img = convert_to_base64(predicted_density_map)
        gt_img = convert_to_base64(gt_density_map) if mat_file else ""
        uploaded_img = convert_to_base64(img_array)

        crowd_count = np.sum(predicted_density_map)
        return render_template('result.html',
                               crowd_count=crowd_count,
                               uploaded_img=uploaded_img,
                               gt_img=gt_img,
                               pred_img=pred_img)

def convert_to_base64(img_array):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
