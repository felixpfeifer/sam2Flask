from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
import os
import numpy as np
from PIL import Image
import cv2
from ai_model import infer_segmentation  # Your AI model function

app = Flask(__name__)

IMAGE_FOLDER = 'static/images/bedroom'

@app.route('/')
def index():
    images = os.listdir(IMAGE_FOLDER)
    images = [img for img in images if img.endswith(('png', 'jpg', 'jpeg'))]  # Filter image files
    current_image = request.args.get('image', images[0])

    if current_image not in images:
        return redirect(url_for('index', image=images[0]))

    return render_template('index.html', images=images, current_image=current_image)

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/segment', methods=['POST'])
def segment_image():
    image_name = request.form['image_name']
    action_type = int(request.form['action_type'])  # 1 for add, 0 for remove
    x_coord = float(request.form['x_coord'])  # Use float instead of int
    y_coord = float(request.form['y_coord'])  # Use float instead of int

    # Assume the frame number is part of the image file name, e.g., '0001.png'
    frame_number:str = image_name.split('.')[0]

    # Call the AI model inference
    mask = infer_segmentation(frame_number, x_coord, y_coord, action_type)

    # Load the original image
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    original_image = cv2.imread(image_path)

    # Convert the mask to a 3-channel image
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Overlay the mask on the original image
    overlayed_image = cv2.addWeighted(original_image, 0.7, mask_3channel, 0.3, 0)

    # Save the overlayed image to return it to the user
    overlay_image_name = f"overlay_{image_name}"
    overlay_image_path = os.path.join(IMAGE_FOLDER, overlay_image_name)
    cv2.imwrite(overlay_image_path, overlayed_image)

    return jsonify({"overlay_image": overlay_image_name})

if __name__ == '__main__':
    app.run(debug=True)
