import os
import argparse
import random
from PIL import Image
from tqdm import tqdm
import shutil
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='../data/generated', help='Directory with raw dataset')
parser.add_argument('--output_dir', default ='../data/test/stylenat_256', help='Directory to save resize dataset')
parser.add_argument('--count', default ='3500', help='Number of Images')
parser.add_argument('--image_name', default ='stylenat', help='Image Name')
parser.add_argument('--image_size', default ='256', help='Output image size')


def convert_image_to_array(filename, img_height=256, img_width=256):
    image = Image.open(filename)
    image = image.resize((img_height, img_width), Image.BILINEAR)
    data = np.asarray(image)
    return data

# Process and resizing images to expected image input size
if __name__ == '__main__':

    args = parser.parse_args()
    data_dir = args.dataset_dir
    output_dir = args.output_dir
    data_dir = args.dataset_dir
    count = int(args.count)
    img_name = args.image_name
    img_size = int(args.image_size)

    i = 0
    for f in tqdm(os.listdir(data_dir)):
        original_file = os.path.join(data_dir, f)
        if original_file.endswith('.png'):
            image = Image.open(original_file)
            image = image.resize((img_size, img_size), Image.BILINEAR)
            dest_file = os.path.join(output_dir, img_name + str(i ) +'.png' )
            image.save(dest_file)
            i += 1
        if i == count:
            break