import os
import numpy as np
from PIL import Image

def convert_image_to_array(filename, img_height=256, img_width=256):
    image = Image.open(filename)
    image = image.resize()
    data = np.asarray(image, (img_height, img_width), Image.BILINEAR)
    return data

def load_data(dir, is_synthetic=True):
    filenames = [os.path.join(dir, f) for f in os.listdir(dir)]
    X = np.array([convert_image_to_array(f) for f in filenames])
    X = X / 255
    return X