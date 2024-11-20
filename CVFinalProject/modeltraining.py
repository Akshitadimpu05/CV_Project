import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from scipy.io import loadmat
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical


# Function to load dataset
def load_dataset(directory):
    images = []
    labels = []

    # Iterate through all .mat files in the ground_truth sub-directory
    for file_name in os.listdir(os.path.join(directory, "ground_truth")):
        if file_name.endswith('.mat'):
            mat_path = os.path.join(directory, "ground_truth", file_name)
            mat_data = loadmat(mat_path)

            # Debug the structure of the mat file to ensure the correct indexing
            print(f"Processing {file_name}")
            print("Keys in mat file:", mat_data.keys())
            print("Image info structure:", mat_data['image_info'])

            try:
                # Dynamically extract label
                image_info = mat_data['image_info'][0][0]  # Extract the main info (this might vary)
                print("Image Info:", image_info)

                # You may need to adjust the label extraction based on the structure of `image_info`
                # Here, I'm assuming that the second element (index 1) holds the label
                label = image_info[1][0]  # Adjust this based on the actual structure
                print("Extracted Label:", label)
            except Exception as e:
                print(f"Error extracting label for {file_name}: {e}")
                continue

            # Locate the corresponding image file (assuming the image name is derived from the mat file name)
            image_name = file_name.replace('GT_', '').replace('.mat', '.jpg')  # Adjust as needed
            image_path = os.path.join(directory, image_name)

            if os.path.exists(image_path):
                # Load and preprocess the image
                image = load_img(image_path, target_size=(224, 224))
                image = img_to_array(image) / 255.0  # Normalize pixel values
                images.append(image)
                labels.append(label)
            else:
                print(f"Image file not found: {image_name}")

    if len(images) == 0 or len(labels) == 0:
        raise ValueError("Dataset is empty or invalid.")

    # Convert to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Return images and labels (categorically encoded)
    return images, to_categorical(labels)

# Example usage:
train_images, train_labels = load_dataset('C:/Users/saiak/Downloads/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data')
val_images, val_labels = load_dataset('C:/Users/saiak/Downloads/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data')

# Function to build the model
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),  # Explicitly define input shape
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Output is a single value for crowd count
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

# Normalize the images to scale pixel values to [0, 1]
train_images = train_images / 255.0
val_images = val_images / 255.0

# Check the shapes
print(f"Train Images Shape: {train_images.shape}")
print(f"Train Labels Shape: {train_labels.shape}")
print(f"Validation Images Shape: {val_images.shape}")
print(f"Validation Labels Shape: {val_labels.shape}")

# Function to train the model
def train_model(model, train_images, train_labels, val_images, val_labels, epochs, batch_size):
    model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size
    )

# Main script to train and save the model
if __name__ == "__main__":
    # Paths to the dataset
    PART_A_TRAIN = 'C:/Users/saiak/Downloads/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data'
    PART_A_TEST = 'C:/Users/saiak/Downloads/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data'

    # Training setup
    print("Loading training data...")
    train_images, train_labels = load_dataset(PART_A_TRAIN)
    print("Loading validation data...")
    val_images, val_labels = load_dataset(PART_A_TEST)

    # Normalize the images
    train_images = train_images / 255.0
    val_images = val_images / 255.0

    # Build the model
    print("Building the model...")
    model = build_model(input_shape=(224, 224, 3))

    # Train the model
    print("Training the model...")
    train_model(model, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=32)

    # Save the trained model
    model.save('crowd_density_model.h5')
    print("Model saved as 'crowd_density_model.h5'")
