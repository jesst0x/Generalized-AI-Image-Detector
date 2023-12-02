import argparse
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import util

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default ='10', help='Directory to save resize dataset')
parser.add_argument('--logging_dir', default ='experiment', help='Directory to save experiment result')
parser.add_argument('--real_train_dir', default ='../data/celeba_256,../data/ffhq_256', help='Directory of real images')
parser.add_argument('--synthetic_train_dir', default ='../data/stylegan_256,../data/progan_256', help='Directory of synthetic images')
parser.add_argument('--real_eval_dir', default ='../data/celeba_256,../data/ffhq_256', help='Directory of real images')
parser.add_argument('--synthetic_eval_dir', default ='../data/stylegan_256,../data/progan_256', help='Directory of synthetic images')
parser.add_argument('--ratio', default ='8', help='Split ratio of training data')
parser.add_argument('--batch_size', default ='5', help='Mini-batch size')


def build_model():
    basemodel = ResNet50(include_top=False, weights='imagenet', pooling='max')
    
    x = basemodel.output
    # Add fully connected layer
    x = layers.Dense(1024, activation='relu')(x)
    # Logistic output layer
    predictions = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=basemodel.input, outputs=predictions)
    
    # Freeze first 45 layers of pre-trained RESNET50
    for layer in basemodel.layers[:45]:
        layer.trainable = False
    for layer in basemodel.layers[45:]:
        layer.trainable = True
    
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model

if __name__ == '__main__':
    args = parser.parse_args()
    num_epoch = int(args.epoch)
    logging_dir = args.logging_dir
    real_train_img = args.real_train_dir.split(',')
    synthetic_train_img = args.synthetic_train_dir.split(',')
    real_eval_img = args.real_eval_dir.split(',')
    synthetic_eval_img = args.synthetic_eval_dir.split(',')
    ratio = int(args.ratio) / 10
    batch_size = int(args.batch_size)
    
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    
    # Random seed for reproducible result
    np.random.seed(23)
    
    # Load and split dataset
    print('Loading data ...........')
    X_train, Y_train = util.combine_dataset_3(real_train_img, synthetic_train_img)
    X_eval, Y_eval = util.combine_dataset_3(real_eval_img, synthetic_eval_img)
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
    model = build_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(X_train, Y_train, epochs=num_epoch, batch_size=batch_size, callbacks =[cp_callback],validation_data=(X_eval, Y_eval), shuffle=True)   
    json.dump(history.history, open(os.path.join(logging_dir, 'result.json'), 'w'))
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