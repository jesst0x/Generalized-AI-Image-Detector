import argparse
import os
import json
import numpy as np
import util


parser = argparse.ArgumentParser()
# parser.add_argument('--real_dir', default ='../data/ffhq', help='Directory of real images')
# parser.add_argument('--synthetic_dir', default ='../data/stylegan,../data/progan', help='Directory of synthetic images')
parser.add_argument('--img_dir', default ='../data/stylegan_256', help='Directory of images')
parser.add_argument('--output_dir', default ='data/stylegan_256', help='Directory of output csv file')
parser.add_argument('--file_name', default ='stylegan_256_train.csv', help='CSV file name')

# Method to convert images to px and save in csv file
if __name__ == '__main__':
    args = parser.parse_args()
    img_dir = args.img_dir
    output_dir = args.output_dir
    filename = args.file_name
    
    if not os.path.exists(output_dir):   
        os.mkdir(output_dir)
    
    X = util.load_data(img_dir)
    print(X.shape)
    np.savetxt(os.path.join(output_dir, filename), X, delimiter=',')
    