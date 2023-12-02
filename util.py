import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

def convert_image_to_array(filename, img_height=256, img_width=256):
    image = Image.open(filename)
    # image = image.resize((img_height, img_width), Image.BILINEAR)
    data = np.asarray(image)
    return data

def load_data(dir):
    print('Loading ' + dir)
    count = 0
    filenames = [os.path.join(dir, f) for f in os.listdir(dir)]
    X = np.array([convert_image_to_array(f) for f in tqdm(filenames)])
    X = X / 255
    print('Completed loading ' + dir)
    return X

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
    
def plot_history(history_dict, save_dir):
    plt.plot(history_dict['accuracy'], label='train_accuracy')
    plt.plot(history_dict['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'accuracy_graph.png'))
    plt.show()
    
    plt.figure()
    plt.plot(history_dict['loss'], label='train_loss')
    plt.plot(history_dict['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'loss_graph.png'))
    plt.show()
    
    plt.figure()
    plt.plot(history_dict['precision'], label='train_precision')
    plt.plot(history_dict['val_precision'], label = 'val_precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'precision_graph.png'))
    plt.show()
    
    plt.figure()
    plt.plot(history_dict['recall'], label='train_recall')
    plt.plot(history_dict['val_recall'], label = 'val_recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'recall_graph.png'))
    plt.show()

def split_dataset(X, label=True, ratio=[0.7, 0.2, 0.1]):
    n = X.shape[0]
    print(X.shape)
    train_size = int(ratio[0] * n)
    eval_size = int(ratio[1] * n)
    test_size = int(ratio[2] * n)
    
    np.random.shuffle(X)
    
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

# Create dataset group - combining different dataset (eg. StyleGAN and ProGAN) into one group to feed into training model
def combine_dataset(real_dir=[], synthetic_dir=[]):
    x_real = [load_data(img_dir) for img_dir in real_dir]
    x_synthetic = [load_data(img_dir) for img_dir in synthetic_dir]
    
    X = np.concatenate(x_synthetic + x_real)
    y_array = []
    n_synthetic = sum([s.shape[0] for s in x_synthetic])
    n_real = sum([r.shape[0] for r in x_real])
    
    if n_synthetic:
        y_array.append(np.ones((n_synthetic, 1)))
    if n_real:
        y_array.append(np.zeros((n_real, 1)))
    
    Y = np.concatenate(y_array)
    print(X.shape)
    print(Y.shape)
    return X, Y


if __name__ == '__main__':
    with open('./experiments/group1/history.json', 'r') as f:
        plot_history(json.load(f), './experiments/group1')