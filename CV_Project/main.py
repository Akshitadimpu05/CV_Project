import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, render_template
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D, ReLU
from tensorflow.keras.models import Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('Agg')

app = Flask(__name__)

# Load VGG16 model and build custom model
input_layer = Input(shape=(224, 224, 3))
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=input_layer)
x = vgg.output
x = Conv2D(512, (3, 3), padding='same', dilation_rate=2, activation='relu')(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = UpSampling2D(size=(8, 8))(x)
output_layer = Conv2D(1, (1, 1), padding='same')(x)
output_layer = ReLU()(output_layer)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file).convert('RGB')
    image_resized = image.resize((224, 224))

    # Automatically detect heads and generate head coordinates
    head_coords = detect_heads(image_resized)

    # Save head coordinates to a .mat file
    save_path = "generated_GT.mat"
    savemat(save_path, {"image_info": [head_coords]})

    # Generate ground truth density map
    density_map_gt = generate_density_map(image, head_coords)

    # Encode ground truth density map as base64
    gt_buf = io.BytesIO()
    plt.imshow(density_map_gt, cmap='jet')
    plt.axis('off')
    plt.savefig(gt_buf, format='png')
    plt.close()
    gt_buf.seek(0)
    gt_img_str = base64.b64encode(gt_buf.read()).decode('utf-8')

    # Encode uploaded image as base64
    img_buf = io.BytesIO()
    image.save(img_buf, format='PNG')
    img_buf.seek(0)
    uploaded_img_str = base64.b64encode(img_buf.read()).decode('utf-8')

    # Preprocess image and predict density map
    image_input = np.expand_dims(np.array(image_resized) / 255.0, axis=0)
    predicted_density_map = model.predict(image_input)
    predicted_density_map = np.maximum(predicted_density_map, 0)
    predicted_density_map = np.squeeze(predicted_density_map)
    crowd_count = np.sum(predicted_density_map)

    # Encode predicted density map as base64
    pred_buf = io.BytesIO()
    plt.imshow(predicted_density_map, cmap='jet')
    plt.axis('off')
    plt.savefig(pred_buf, format='png')
    plt.close()
    pred_buf.seek(0)
    pred_img_str = base64.b64encode(pred_buf.read()).decode('utf-8')

    # Render the result template with all images and count
    return render_template(
        'result.html',
        uploaded_img=uploaded_img_str,
        gt_img=gt_img_str,
        pred_img=pred_img_str,
        crowd_count=crowd_count
    )

def detect_heads(image):
    # Placeholder function - replace with head detection logic
    head_coords = np.array([[50, 50], [100, 100], [150, 150]])
    return head_coords

def generate_density_map(image, head_coords, sigma=4):
    density_map = np.zeros((image.height, image.width))
    for coord in head_coords:
        x, y = min(int(coord[0]), image.width - 1), min(int(coord[1]), image.height - 1)
        density_map[y, x] = 1
    density_map = gaussian_filter(density_map, sigma=sigma)
    return density_map

if __name__ == '__main__':
    app.run(debug=True)