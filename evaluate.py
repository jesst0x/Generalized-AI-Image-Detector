import util
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import tensorflow as tf
import shutil
import model
    
    
parser = argparse.ArgumentParser()

parser.add_argument('--real_test_dir', default ='../data/eval/celeba_256,../data/eval/ffhq_256', help='Directory of real images')
parser.add_argument('--synthetic_test_dir', default ='../data/eval/stylegan_256,../data/eval/progan_256', help='Directory of synthetic images')
parser.add_argument('--logging_dir', default='experiments/group1', help='Directory to save evaluation result')
parser.add_argument('--file_name', default='test_result.json', help='Directory to save evaluation result')
parser.add_argument('--checkpoint_path', default='experiments/group1/training_checkpoints/cp-0004.ckpt', help='File to trained model weight')

# Evaluate a test set from a trained model
if __name__ == '__main__':
    args = parser.parse_args()
    logging_dir = args.logging_dir
    file_name = args.file_name
    real_test_img = args.real_test_dir.split(',')
    synthetic_test_img = args.synthetic_test_dir.split(',')
    checkpoint_path = args.checkpoint_path
    
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    
    X_test, Y_test = util.combine_dataset(real_test_img, synthetic_test_img)
    X_test = tf.keras.applications.resnet50.preprocess_input(X_test)
    
    model = model.build_model()
    model.load_weights(checkpoint_path)
    
    y_pred = model.predict(X_test).ravel()
    loss, acc, auc, precision, recall = model.evaluate(X_test, Y_test)
    result = {'loss': loss, 'accuracy': acc, 'auc': auc, 'precision': precision, 'recall': recall}
    json.dump(result, open(os.path.join(logging_dir, file_name), 'w'))
    print(result)