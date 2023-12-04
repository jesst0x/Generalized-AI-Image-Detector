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

import util

parser = argparse.ArgumentParser()

parser.add_argument('--logging_dir', default ='experiments/group2/l40_a6_b20_n40', help='Directory to save experiment result')

parser.add_argument('--synthetic_train_dir', default ='../data/train/stylegan_256,../data/train/progan_256', help='Directory of synthetic images')
parser.add_argument('--synthetic_eval_dir', default ='../data/eval/stylegan2_256', help='Directory of synthetic images')
parser.add_argument('--real_train_dir', default ='../data/train/celeba_256,../data/train/ffhq_256', help='Directory of real images')
parser.add_argument('--real_eval_dir', default ='../data/eval/celeba_256,../data/eval/ffhq_256', help='Directory of real images')
parser.add_argument('--synthetic_train_n', default ='1750,1750', help='Number of training example in each synthetic train set separated by comma')
parser.add_argument('--synthetic_eval_n', default ='1000', help='Number of training example in each synthetic eval separated by comma')
parser.add_argument('--real_train_n', default ='1750,1750', help='Number of training example in each real train set separated by comma')
parser.add_argument('--real_eval_n', default ='500,500', help='Number of training example in each real eval set separated by comma')

parser.add_argument('--batch_size', default ='20', help='Mini-batch size')
parser.add_argument('--epoch', default ='40', help='Directory to save resize dataset')
parser.add_argument('--learning_rate', default='1e-6', help='Learning rate')
parser.add_argument('--freeze_layer', default='40', help='Frozen first n resnet50 layer')


def build_model(learning_rate=1e-5, freeze_layer=45):
    basemodel = ResNet50(include_top=False, weights='imagenet', pooling='max')
    
    x = basemodel.output
    # Add fully connected layer
    x = layers.Dense(1024, activation='relu')(x)
    # Logistic output layer
    predictions = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=basemodel.input, outputs=predictions)
    
    # Freeze first 45 layers of pre-trained RESNET50
    for layer in basemodel.layers[:freeze_layer]:
        layer.trainable = False
    for layer in basemodel.layers[freeze_layer:]:
        layer.trainable = True
    
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model

if __name__ == '__main__':
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
    # Hyperparameters
    num_epoch = int(args.epoch)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    freeze_layer = int(args.freeze_layer)
    # checkpoint_path = args.checkpoint_path
    
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    
    # Random seed for reproducible result
    np.random.seed(23)
    
    # Load and split dataset
    print('Loading data ...........')
    X_train, Y_train = util.combine_dataset(real_train_img, synthetic_train_img, real_train_n, synthetic_train_n)
    X_eval, Y_eval = util.combine_dataset(real_eval_img, synthetic_eval_img, real_eval_n, synthetic_eval_n)
    print('Dataset loaded!')
    
    # Converting RGB to BGR to be expected input to RESNET50 
    X_train = tf.keras.applications.resnet50.preprocess_input(X_train)
    X_eval = tf.keras.applications.resnet50.preprocess_input(X_eval)
    
    # Set up training checkpoint to save weights for prediction later
    checkpoint_dir = os.path.join(logging_dir, 'training_checkpoints')
    if not os.path.exists(checkpoint_dir):   
        os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
    n_batches = len(X_train) // batch_size
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True, save_freq=2 * n_batches) 
     
    # Fit ResNet50 model with transfer learning
    print('******Training model*****')
    model = build_model(learning_rate, freeze_layer)
    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(X_train, Y_train, epochs=num_epoch, batch_size=batch_size, callbacks =[cp_callback],validation_data=(X_eval, Y_eval), shuffle=True)   
    json.dump(history.history, open(os.path.join(logging_dir, 'history.json'), 'w'))
    print('Training completed!')
    print('Training result', history.history)
    
    # Plot loss and accuracy graph
    util.plot_history(history.history, logging_dir)
    
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
