import argparse
import os
import json
import numpy as np
import util


parser = argparse.ArgumentParser()
parser.add_argument('--real_dir', default ='../data/celeba,../data/ffhq', help='Directory of real images')
parser.add_argument('--synthetic_dir', default ='../data/stylegan,../data/progan', help='Directory of synthetic images')


if __name__ == '__main__':
    args = parser.parse_args()
    real_img_dir = args.real_dir.split(',')
    synthetic_img_dir = args.synthetic_dir.split(',')
    
    X = util.load_data('../data/ffhq_256')
    print(X.shape)
    
    # for i, img_dir in enumerate(synthetic_img_dir):
    #     np.savez('synthetic' + str(i) + '.npz', util.load_data(img_dir))

    # for i, img_dir in enumerate(real_img_dir):
    #     np.savez('real' + str(i) + '.npz', util.load_data(img_dir))
    # np.savez('synthetic1.npz', util.load_data('../data/progan'))
    # Load dataset
    # with open('train_synthetic.npy', 'wb') as f:
    #     for img_dir in synthetic_img_dir:
    #         np.save(f, util.load_data(img_dir))

    
    # with open('train_real.npy', 'wb') as f:
    #     for img_dir in real_img_dir:
    #         np.save(f, util.load_data(img_dir))

    # array = util.load_data('../data/test')
    # dest = 'gs://cs229_jt/test.npy'
    # np.save(file_io.FileIO(dest, 'w'), array)
    
    # f = BytesIO(file_io.read_file_to_string(dest, binary_mode=True))
    # arr = np.load(f)
    # print(arr.shape)
    

