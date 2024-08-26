import numpy as np
from PIL import Image

def infer_segmentation(frame_number:str, x, y, action):
    # Example function to simulate AI inference
    # Here, you'd load the frame (image) and apply your model

    # Create a dummy mask (this should be your model's output)
    image = Image.open(f'static/images/bedroom/{frame_number}.jpg')
    image = np.array(image)
    h,w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)  # Assume 640x480 image size
    # convert the x,y float coordinates to int
    x = int(x)
    y = int(y)
    if action == 1:
        # Simulate adding a point to the mask
        # A small square for simplicity
        mask[y - 5:y + 5, x - 5:x + 5] = 255
    else:
        # Simulate removing a point from the mask
        mask[y - 5:y + 5, x - 5:x + 5] = 0

    return mask  # This should be the actual mask from your model
