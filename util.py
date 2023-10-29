import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_image_to_array(filename, img_height=256, img_width=256):
    image = Image.open(filename)
    # image = image.resize((img_height, img_width), Image.BILINEAR)
    data = np.asarray(image)
    return data

def load_data(dir):
    print('Loading ' + dir)
    count = 0
    filenames = []
    for f in tqdm(os.listdir(dir)):
        filenames.append(os.path.join(dir, f))
        count += 1
        if count == 2500:
            break
    # filenames = [os.path.join(dir, f) for f in os.listdir(dir)]
    X = np.array([convert_image_to_array(f) for f in tqdm(filenames)])
    X = X / 255
    print('Completed loading ' + dir)
    return X