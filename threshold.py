import argparse
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import model

import util
import evaluate

parser = argparse.ArgumentParser()

parser.add_argument('--logging_dir', default ='experiments/group3/l45_a6_b10_n20', help='Directory to save experiment result')
parser.add_argument('--real_eval_dir', default ='../data/eval/celeba_256,../data/eval/ffhq_256', help='Directory of real images')
parser.add_argument('--synthetic_eval_dir', default ='../data/eval/stylegan_256', help='Directory of synthetic images')
parser.add_argument('--real_eval_n', default ='500,500', help='Number of training example in each real eval set separated by comma')
parser.add_argument('--synthetic_eval_n', default ='1000', help='Number of training example in each synthetic eval separated by comma')
parser.add_argument('--checkpoint_path', default='experiments/group3/l45_a6_b10_n20/training_checkpoints/cp-0004.ckpt', help='File to trained model weight')

# Find optimal threshold by using evaluation set and saved weight from training checkpoint
if __name__ == '__main__':
    args = parser.parse_args()
    logging_dir = args.logging_dir
    real_eval_img = args.real_eval_dir.split(',')
    synthetic_eval_img = args.synthetic_eval_dir.split(',')
    synthetic_eval_n = [int(i) for i in args.synthetic_eval_n.split(',')]
    real_eval_n = [int(i) for i in args.real_eval_n.split(',')]
    checkpoint_path = args.checkpoint_path    
    
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    
    # Random seed for reproducible result
    np.random.seed(23)
    
    # Load and split dataset
    print('Loading data ...........')
    X_eval, Y_eval = util.combine_dataset(real_eval_img, synthetic_eval_img, real_eval_n, synthetic_eval_n)
    X_eval = tf.keras.applications.resnet50.preprocess_input(X_eval)
    
    # Create model instance and loaded trained weight 
    model = model.build_model()
    model.load_weights(checkpoint_path)
    
    ## Compute Optimal thresold using argmax f1 score
    Y_eval_pred = model.predict(X_eval).ravel()
    precision, recall, thresolds = precision_recall_curve(Y_eval, Y_eval_pred)
    f1scores = 2 * (precision * recall) / (precision + recall)
    idx = np.argmax(f1scores)
    optimal_threshold = thresolds[idx]
    print('Optimal threshold: ', optimal_threshold, 'Max f1 scores: ', f1scores[idx])
    
    # Getting metrics of evaluatin set using optimal threshold
    result = evaluate.compute_metrics(Y_eval, Y_eval_pred, optimal_threshold)
    print(result)