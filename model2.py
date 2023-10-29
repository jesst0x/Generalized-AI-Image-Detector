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
parser.add_argument('--logging_dir', default ='result', help='Directory to save experiment result')
parser.add_argument('--real_dir', default ='../data/celeba_256,../data/ffhq_256', help='Directory of real images')
parser.add_argument('--synthetic_dir', default ='../data/stylegan_256,../data/progan_256', help='Directory of synthetic images')


def build_model():
    basemodel = ResNet50(include_top=False, weights='imagenet', pooling='max')
    
    x = basemodel.output
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=basemodel.input, outputs=predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model

# def load_data(data_dir, img_height, img_width):
#     X = tf.keras.utils.image_dataset_from_directory(data_dir, seed=23, image_size=(img_height, img_width))
#     normalization_layer = tf.keras.layers.Rescaling(1./255)
#     normalized_X = X.map(lambda x: normalization_layer(x))
    
#     return normalized_X

def split_dataset(X, label=True, ratio=[0.7, 0.2, 0.1]):
    print(X.shape)
    n = X.shape[0]
    train_size = int(ratio[0] * n)
    eval_size = int(ratio[1] * n)
    test_size = int(ratio[2] * n)
    
    np.random.shuffle(X)
    
    # X = tf.random.shuffle(X, seed=23)
    # X_train = X.take(train_size)
    # X_test = X.skip(train_size)
    # X_eval = X_test.take(eval_size)
    # X_test = X_test.skip(eval_size)
    
    X_train = X[:train_size, :, :, :]
    X_eval = X[train_size:train_size + eval_size, :, :, :]
    X_test = X[-test_size:, :, :, :]
    
    if label:
        Y_train = np.ones((X_train.shape[0], 1))
        Y_test = np.ones((X_test.shape[0], 1))
        Y_eval = np.ones((X_eval.shape[0], 1))
    else:
        Y_train = np.zeros((X_train.shape[0], 1))
        Y_test = np.zeros((X_test.shape[0], 1))
        Y_eval = np.zeros((X_eval.shape[0], 1))
    
    return (X_train, X_eval, X_test, Y_train, Y_eval, Y_test)
    
def combine_dataset(real=[], synthetic=[]):
    X_train, X_eval, X_test, Y_train, Y_eval, Y_test = real[0]
    
    for i in range(1, len(real)):
        x_train, x_eval, x_test, y_train, y_eval, y_test = real[i]
        X_train = np.concatenate((X_train, x_train))
        X_eval = np.concatenate((X_eval, x_eval))
        X_test = np.concatenate((X_test, x_test))
        
        Y_train = np.concatenate((Y_train, y_train))
        Y_eval = np.concatenate((Y_eval, y_eval))
        Y_test = np.concatenate((Y_test, y_test))
    
    for x_train, x_eval, x_test, y_train, y_eval, y_test in synthetic:
        X_train = np.concatenate((X_train, x_train))
        X_eval = np.concatenate((X_eval, x_eval))
        X_test = np.concatenate((X_test, x_test))
        
        Y_train = np.concatenate((Y_train, y_train))
        Y_eval = np.concatenate((Y_eval, y_eval))
        Y_test = np.concatenate((Y_test, y_test))

    return X_train, X_eval, X_test, Y_train, Y_eval, Y_test

def plot_roc_curve(tpr, fpr, auc, title, path):
    plt.figure()
    plt.plot([0, 1], [0, 1], ':')
    plt.plot(fpr, tpr, label='Auc = {:.3f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(path)
    plt.show()
    
def plot_history(history_dict, logging_dir):
    plt.plot(history_dict['accuracy'], label='train_accuracy')
    plt.plot(history_dict['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(logging_dir, 'accuracy_graph.png'))
    plt.show()
    
    plt.figure()
    plt.plot(history_dict['loss'], label='train_loss')
    plt.plot(history_dict['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(logging_dir, 'loss_graph.png'))
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    num_epoch = int(args.epoch)
    logging_dir = args.logging_dir
    real_img_dir = args.real_dir.split(',')
    synthetic_img_dir = args.synthetic_dir.split(',')
    

    X_synthetic = [util.load_data(img_dir) for img_dir in synthetic_img_dir]
    X_real = [util.load_data(img_dir) for img_dir in real_img_dir]
    print(X_real[0].shape)
    X_train, X_eval, X_test, Y_train, Y_eval, Y_test = combine_dataset([split_dataset(x_real, False) for x_real in X_real], [split_dataset(x_synthetic) for x_synthetic in X_synthetic])
    
    print('Dataset loaded')
    
    X_train = tf.keras.applications.resnet50.preprocess_input(X_train)
    X_eval = tf.keras.applications.resnet50.preprocess_input(X_eval)
    X_test = tf.keras.applications.resnet50.preprocess_input(X_test)
    
    print('training model')
    # Fit ResNet50 model
    model = build_model()
    history = model.fit(X_train, Y_train, epochs=num_epoch, batch_size=100, validation_data=(X_eval, Y_eval), shuffle=True)   
    json.dump(history.history, open(os.path.join(logging_dir, 'result.json'), 'w'))
    print('Training completed')
    print(history.history)
    # Plot loss and accuracy graph
    plot_history(history.history, logging_dir)
    
    #Plot ROC Curve
    Y_eval_pred = model.predict(X_eval).ravel()
    fpr_eval, tpr_eval, thresolds_eval = roc_curve(Y_eval, Y_eval_pred)
    auc_eval = auc(fpr_eval, tpr_eval)
    plot_roc_curve(tpr_eval, fpr_eval, auc_eval, 'ROC Curve - Evaluation Set ', os.path.join(logging_dir, 'evaluation_ROC.png'))
    print('Completed')