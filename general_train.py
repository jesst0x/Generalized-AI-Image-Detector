import argparse
import os
import json
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np

import util
import resnet
import ensemble

parser = argparse.ArgumentParser()

# Logging directory
parser.add_argument('--logging_dir', default ='experiments/group3/ensemble_1,1,2,1,1,2_a6_b100_n5', help='Directory to save experiment result')

# Dataset directory
parser.add_argument('--synthetic_train_dir', default ='../data/train/stylegan_256,../data/train/progan_256,../data/train/vqgan_256,../data/train/ldm_256', help='Directory of synthetic images')
parser.add_argument('--synthetic_eval_dir', default ='../data/eval/stylegan2_256', help='Directory of synthetic images')
parser.add_argument('--real_train_dir', default ='../data/train/celeba_256,../data/train/ffhq_256', help='Directory of real images')
parser.add_argument('--real_eval_dir', default ='../data/eval/celeba_256,../data/eval/ffhq_256', help='Directory of real images')
parser.add_argument('--synthetic_train_n', default ='850,850,850,850', help='Number of training example in each synthetic train set separated by comma')
parser.add_argument('--synthetic_eval_n', default ='900', help='Number of training example in each synthetic eval separated by comma')
parser.add_argument('--real_train_n', default ='1700,1700', help='Number of training example in each real train set separated by comma')
parser.add_argument('--real_eval_n', default ='450,450', help='Number of training example in each real eval set separated by comma')

# Hyperparameters
parser.add_argument('--batch_size', default ='100', help='Mini-batch size')
parser.add_argument('--epoch', default ='5', help='Directory to save resize dataset')
parser.add_argument('--learning_rate', default='1e-6', help='Learning rate')
parser.add_argument('--freeze_layer', default='40', help='Frozen first n resnet50 layer')
parser.add_argument('--nn_layers', default='1,1,2,1,1,2', help='Number of layers in base estimators of ensemble model')
# Model type
parser.add_argument('--is_resnet', default='y', help='Resnet or ensemble model')


# Training resnet or ensemble model
def train():
    args = parser.parse_args()
    logging_dir = args.logging_dir
    real_train_img = args.real_train_dir.split(',')
    synthetic_train_img = args.synthetic_train_dir.split(',')
    real_eval_img = args.real_eval_dir.split(',')
    synthetic_eval_img = args.synthetic_eval_dir.split(',')
    synthetic_train_n = [int(i) for i in args.synthetic_train_n.split(',')]
    real_train_n = [int(i) for i in args.real_train_n.split(',')]
    synthetic_eval_n = [int(i) for i in args.synthetic_eval_n.split(',')]
    real_eval_n = [int(i) for i in args.real_eval_n.split(',')]
    # Resnet or ensemble model
    is_resnet = True if args.is_resnet == 'y' else False
    # Hyperparameters
    num_epoch = int(args.epoch)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    freeze_layer = int(args.freeze_layer)
    nn_layers = [int(l) for l in args.nn_layers.split(',')]

    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
        
    # Save model architects
    model_summary = {'learning_rate': learning_rate, 'batch_size':batch_size, 'num_epoch': num_epoch}
    if is_resnet:
        model_summary['model']  = 'resnet'  
        model_summary['freeze_layer'] = freeze_layer
    else:
        model_summary['model'] = 'ensemble'
        model_summary['layers'] = nn_layers
    json.dump(model_summary, open(os.path.join(logging_dir, 'model_summary.json'), 'w'))
    
    # Random seed for reproducible result
    np.random.seed(23)
    
    # Load and split dataset
    print('Loading data ...........')
    X_train, Y_train = util.combine_dataset(real_train_img, synthetic_train_img, real_train_n, synthetic_train_n)
    X_eval, Y_eval = util.combine_dataset(real_eval_img, synthetic_eval_img, real_eval_n, synthetic_eval_n)
    print('Dataset loaded!')
    
    # Converting RGB to BGR to be expected input to RESNET50 
    if is_resnet:
        X_train = tf.keras.applications.resnet50.preprocess_input(X_train)
        X_eval = tf.keras.applications.resnet50.preprocess_input(X_eval)
        
    model = None
    if is_resnet:
        model = resnet.train(X_train, Y_train, X_eval, Y_eval, num_epoch, batch_size, learning_rate, freeze_layer, logging_dir)
    else:
        model = ensemble.AdaBoost()
        model.train(X_train, Y_train, X_eval, Y_eval, learning_rate, batch_size, num_epoch, nn_layers, logging_dir)
    
    # Plotting ROC curve for evaluation set
    Y_eval_pred = model.predict(X_eval).ravel()
    fpr_eval, tpr_eval, thresolds_eval = roc_curve(Y_eval, Y_eval_pred)
    auc_eval = auc(fpr_eval, tpr_eval)
    util.plot_roc_curve(tpr_eval, fpr_eval, auc_eval, 'ROC Curve - Evaluation Set ', os.path.join(logging_dir, 'evaluation_ROC.png'))
    print('Completed')
    
    ## Compute Optimal thresold using armax f1 score
    precision, recall, thresolds = precision_recall_curve(Y_eval, Y_eval_pred)
    f1scores = 2 * (precision * recall) / (precision + recall)
    idx = np.argmax(f1scores)
    optimal_threshold = thresolds[idx]
    print(optimal_threshold, f1scores[idx])

if __name__ == '__main__':
    train()
  