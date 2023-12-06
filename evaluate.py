import util
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import tensorflow as tf
import shutil
import model
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
    
    
parser = argparse.ArgumentParser()

parser.add_argument('--real_test_dir', default ='../data/eval/celeba_256,../data/eval/ffhq_256', help='Directory of real images')
parser.add_argument('--real_test_n', default ='250,250', help='Number of training example in each real test sets separated by comma')
parser.add_argument('--synthetic_test_dir', default ='../data/test/stylegan2_256', help='Directory of synthetic images')
parser.add_argument('--synthetic_test_n', default ='1000', help='Number of training example in each synthetic test set separated by comma')

parser.add_argument('--logging_dir', default='experiments/group1/l45_a5_b10_n10/', help='Directory to save evaluation result')
parser.add_argument('--file_name', default='eval', help='Directory to save evaluation result')

parser.add_argument('--checkpoint_path', default='experiments/group1/l45_a5_b10_n10/training_checkpoints/cp-0004.ckpt', help='File to trained model weight')
parser.add_argument('--threshold', default='0.5', help='Threshold for binary classification')

def compute_metrics(y, y_pred_prob, threshold=0.5):
    n = y.shape[0]
    y_pred = np.where(y_pred_prob > threshold, 1, 0)
    print(y_pred[:10], y_pred[-10:])
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    acc = (tp + tn) / n
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    neg_recall = tn / (tn + fp)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Threshold independent
    fpr_eval, tpr_eval, thresolds_eval = roc_curve(y, y_pred_prob)
    auc_eval = auc(fpr_eval, tpr_eval)
    
    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'neg_recall': neg_recall,
        'f1_score': f1_score,
        'auc': auc_eval,
    }

# Evaluate a test set with weights saved in checkpoints of a trained model.
if __name__ == '__main__':
    args = parser.parse_args()
    logging_dir = args.logging_dir
    file_name = args.file_name
    real_test_img = args.real_test_dir.split(',')
    synthetic_test_img = args.synthetic_test_dir.split(',')
    synthetic_test_n = [int(i) for i in args.synthetic_test_n.split(',')]
    real_test_n = [int(i) for i in args.real_test_n.split(',')]
    checkpoint_path = args.checkpoint_path
    threshold = float(args.threshold) # Optimal threshold in evaluation set
    
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    
    # Load evaluation or test data
    X_test, Y_test = util.combine_dataset(real_test_img, synthetic_test_img, real_test_n, synthetic_test_n)
    X_test = tf.keras.applications.resnet50.preprocess_input(X_test)
    
    # Create model instance and loaded trained weight 
    model = model.build_model()
    model.load_weights(checkpoint_path)
    
    # Predict and compute metrics of prediction
    y_pred_prob = model.predict(X_test)
    result = compute_metrics(Y_test, y_pred_prob, threshold)
    json.dump(result, open(os.path.join(logging_dir, 'result_' + file_name + '_t' + str(threshold) + '.json'), 'w'))
    print(result)