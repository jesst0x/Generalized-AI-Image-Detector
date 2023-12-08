import argparse
import os
import json
import tensorflow as tf
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np

import util
import general_evaluate
import resnet
import ensemble

parser = argparse.ArgumentParser()

parser.add_argument('--logging_dir', default ='experiments/group1/resnet_l40_a5_b20_n20_decay', help='Directory to save experiment result')

# Evaluation set directory
parser.add_argument('--real_eval_dir', default ='../data/eval/celeba_256,../data/eval/ffhq_256', help='Directory of real images')
parser.add_argument('--synthetic_eval_dir', default ='../data/eval/progan_256', help='Directory of synthetic images')
parser.add_argument('--real_eval_n', default ='500,500', help='Number of training example in each real eval set separated by comma')
parser.add_argument('--synthetic_eval_n', default ='1000', help='Number of training example in each synthetic eval separated by comma')
# Directory to load trained weight
parser.add_argument('--model_dir', default='experiments/group1/resnet_l40_a5_b20_n20_decay', help='File to trained model weight')
parser.add_argument('--resnet_checkpoint', default='cp-0008.ckpt', help='File to trained model weight of resnet')


def compute_optimal_threshold(Y, Y_pred_prob, mode='roc'):
    if mode == 'roc':
        fpr, tpr, thresholds = roc_curve(Y, Y_pred_prob)
        return thresholds[np.argmin(np.abs(fpr + tpr - 1))]
    else:
        # f1 score mode
        precision, recall, thresolds = precision_recall_curve(Y, Y_pred_prob)
        f1scores = 2 * (precision * recall) / (precision + recall)
        return thresolds[np.argmax(f1scores)]
    
# Find optimal threshold by using evaluation set and saved weight from training checkpoint    
def main():
    args = parser.parse_args()
    logging_dir = args.logging_dir
    real_eval_img = args.real_eval_dir.split(',')
    synthetic_eval_img = args.synthetic_eval_dir.split(',')
    synthetic_eval_n = [int(i) for i in args.synthetic_eval_n.split(',')]
    real_eval_n = [int(i) for i in args.real_eval_n.split(',')]
    model_dir = args.model_dir      
    checkpoint_path = os.path.join(model_dir, 'training_checkpoints/' + args.resnet_checkpoint)
    
    # Load model details
    with open(os.path.join(model_dir, 'model_summary.json')) as f:
        model_summary = json.load(f)
        print(model_summary)
    
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    
    # Random seed for reproducible result
    np.random.seed(23)
    
    # Load and split dataset
    print('Loading data ...........')
    X_eval, Y_eval = util.combine_dataset(real_eval_img, synthetic_eval_img, real_eval_n, synthetic_eval_n)
    
    # Create model instance and loaded trained weight 
    if model_summary['model'] == 'resnet':
        X_eval = tf.keras.applications.resnet50.preprocess_input(X_eval)
        model = resnet.build_model()
        model.load_weights(checkpoint_path).expect_partial() # We don't need decay learning rate in predict
    else:
        # Ensemble
        with open(os.path.join(model_dir, 'weights.json')) as f:
            weights = json.load(f)
        model = ensemble.AdaBoost()
        model.load_weights(weights)
    
    ## Compute Optimal thresold using argmax f1 score
    Y_eval_pred = model.predict(X_eval).ravel()
    roc_optimal_threshold = compute_optimal_threshold(Y_eval, Y_eval_pred)
    f1_optimal_threshold = compute_optimal_threshold(Y_eval, Y_eval_pred, 'f1')
    threshold_summary = {'roc optimal threshold': float(roc_optimal_threshold), 'f1_optimal_threshold': float(f1_optimal_threshold),'checkpoint': checkpoint_path}
    json.dump(threshold_summary, open(os.path.join(logging_dir, 'threshold.json'), 'w'))
    print('ROC Optimal threshold: ', roc_optimal_threshold, 'f1 Optimal Threshold',f1_optimal_threshold)
    
    # Getting metrics of evaluatin set using optimal threshold
    result_roc = general_evaluate.compute_metrics(Y_eval, Y_eval_pred, roc_optimal_threshold)
    print('Result with roc optimal threshold: ', result_roc)
    result_f1 = general_evaluate.compute_metrics(Y_eval, Y_eval_pred, f1_optimal_threshold)
    print('Result with f1 optimal threshold: ', result_f1)
    

if __name__ == '__main__':
    main()