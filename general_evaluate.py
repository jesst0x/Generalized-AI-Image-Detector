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
import resnet
import ensemble
import PIL.Image
    
    
parser = argparse.ArgumentParser()
# Evaluation or test dataset
parser.add_argument('--real_test_dir', default ='../data/eval/celeba_256,../data/eval/ffhq_256', help='Directory of real images')
parser.add_argument('--real_test_n', default ='500,500', help='Number of training example in each real test sets separated by comma')
parser.add_argument('--synthetic_test_dir', default ='../data/test/ldm_256', help='Directory of synthetic images')
parser.add_argument('--synthetic_test_n', default ='1000', help='Number of training example in each synthetic test set separated by comma')

# Logging directory
parser.add_argument('--logging_dir', default='experiments/group1/resnet_l40_a5_b10_n20/', help='Directory to save evaluation result')
parser.add_argument('--file_name', default='ldm_test', help='Directory to save evaluation result')

# Directory to load trained weight
parser.add_argument('--model_dir', default='experiments/group1/resnet_l40_a5_b10_n20/', help='File to trained model weight')
parser.add_argument('--checkpoint_path', default='experiments/group1/resnet_l40_a5_b10_n20/training_checkpoints/cp-0006.ckpt', help='File to trained model of resnet')

# Threshold to calculate accuracy and confusion matrix
parser.add_argument('--threshold', default='0.77', help='Threshold for binary classification')

# To save misclassified images for error analysis
parser.add_argument('--save_img', default='y', help='To save misclassified images')

# Model type
parser.add_argument('--is_resnet', default='y', help='Resnet or ensemble model')

def compute_metrics(y, y_pred_prob, threshold=0.5):
    n = y.shape[0]
    y_pred = np.where(y_pred_prob > threshold, 1, 0)
    # print(y_pred[:10], y_pred[-10:])
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
    model_dir = args.model_dir
    threshold = float(args.threshold) # Optimal threshold in evaluation set
    save_img = True if args.save_img == 'y' else False
    # Resnet or ensemble model
    is_resnet = True if args.is_resnet == 'y' else False

    with open(os.path.join(model_dir, 'model_summary.json')) as f:
        model_summary = json.load(f)
        print(model_summary)
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    
    # Load evaluation or test data
    X_test, Y_test = util.combine_dataset(real_test_img, synthetic_test_img, real_test_n, synthetic_test_n)
    X_test_processed = X_test.copy()
    
    # Create model instance and loaded trained weight 
    if model_summary['model'] == 'resnet':
        X_test_processed = tf.keras.applications.resnet50.preprocess_input(X_test_processed)
        model = resnet.build_model()
        model.load_weights(checkpoint_path)
    else:
        # Ensemble
        with open(os.path.join(model_dir, 'weights.json')) as f:
            weights = json.load(f)
        model = ensemble.AdaBoost()
        model.load_weights(weights)
    
    # Predict and compute metrics of prediction
    y_pred_prob = model.predict(X_test_processed)
    result = compute_metrics(Y_test, y_pred_prob, threshold)
    json.dump(result, open(os.path.join(logging_dir, 'result_' + file_name + '_t' + str(threshold) + '.json'), 'w'))
    print(result)
    # Save misclassified image index based on threshold for error analysis
    y_pred = np.where(y_pred_prob > threshold, 1, 0)
    misclassified_filter = (y_pred != Y_test).reshape((y_pred.shape[0],))
    index = np.arange(y_pred.shape[0])
    misclassified_index = index[misclassified_filter]
    json.dump({'misclassified': misclassified_index.tolist()}, open(os.path.join(logging_dir, file_name + '_misclassified.json'), 'w'))
    if save_img:
        outdir = os.path.join(logging_dir, file_name + '_misclassfied')
        if not os.path.exists(outdir):   
            os.mkdir(outdir)
        X_misclassified = X_test[misclassified_filter, :, :, :]
        n_x, img_size, img_size, channel = X_misclassified.shape
        for i in range(n_x):
            img_arr = np.uint8(X_misclassified[i, :, :, :] * 255)
            PIL.Image.fromarray(img_arr).save(f'{outdir}/{misclassified_index[i]}.png')