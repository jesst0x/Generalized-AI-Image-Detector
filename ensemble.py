import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import gc

class CNN:
    def __init__(self, n_layer=2):
        self.n_layer = n_layer
        self.model = None
        
    def train(self, X, Y, X_eval, Y_eval, learning_rate, batch_size, n_epoch, sample_weight, logging_dir):
        checkpoint_dir = os.path.join(logging_dir, 'training_checkpoints')
        if not os.path.exists(checkpoint_dir):   
            os.mkdir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
        n_batches = X.shape[0] // batch_size
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True, save_freq=n_batches) 
        
        img_size = X.shape[1]
        # Build model
        self.build_model(img_size, learning_rate)
        history = self.model.fit(X, Y, epochs=n_epoch, batch_size=batch_size, sample_weight=sample_weight,validation_data=(X_eval, Y_eval),callbacks =[cp_callback], shuffle=True) 
        json.dump(history.history, open(os.path.join(logging_dir, 'history.json'), 'w'))
        
        
    def predict(self, X):
        prob = self.predict_prob(X)
        pred = prob > 0.5
        return pred
        
    def predict_prob(self, X):
        batch_size = 500
        n = X.shape[0]
        prob = np.zeros((n, 1))
        n_batch = n // batch_size
        
        # Break into chunks due to memory
        for i in range(n_batch):
            prob[i * batch_size: (i + 1) * batch_size] = self.model.predict(X[i * batch_size: (i + 1) * batch_size])
        if n_batch * batch_size < n:
            prob[n_batch * batch_size:] = self.model.predict(X[n_batch * batch_size:])
        return prob
        
    def build_model(self, img_size=256, learning_rate=1e-5):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        for i in range(1, self.n_layer):
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        self.model = model
        return model

class AdaBoost:
    def __init__(self):
        self.alpha = None # Data weight
        self.estimator_weights = [] # Estimator weight, w
        self.estimator_dirs = [] # Due to memory, save directory instead of model
        self.weighted_errors = []
        self.nn_layers = []
        
    def clear_memory(self, cnn):
        tf.keras.backend.clear_session()
        del cnn.model
        del cnn
        gc.collect()
        
    def train(self, X, Y, X_eval, Y_eval, learning_rate, batch_size, n_epoch, nn_layers, logging_dir):
        n = X.shape[0]
        self.alpha = np.ones((n, 1)) / n
        self.nn_layers = nn_layers
        
        for i, layer in enumerate(nn_layers):
            # Save checkpoint directory path
            new_logging_dir = os.path.join(logging_dir, str(i)+ '_' + str(layer))
            self.estimator_dirs.append(new_logging_dir)
            if not os.path.exists(new_logging_dir):   
                os.mkdir(new_logging_dir)
            
            print('layer', i, new_logging_dir)
            cnn = CNN(layer)
            cnn.train(X, Y, X_eval, Y_eval, learning_rate, batch_size, n_epoch, self.alpha, new_logging_dir)
            Y_pred = (cnn.predict(X)).reshape((n, 1))
            self.update_param(Y, Y_pred)
            
            # Release previous model from memory
            del Y_pred
            self.clear_memory(cnn)

        result = {'estimator_weights': self.estimator_weights, 'weighted_errors':self.weighted_errors, 'estimator_dir': self.estimator_dirs, 'nn_layers': self.nn_layers}
        print(result)
        json.dump(result, open(os.path.join(logging_dir, 'weights.json'), 'w'))
            
    def update_param(self, Y, Y_pred):
        weighted_error = self.estimate_error(Y, Y_pred)
        self.weighted_errors.append(weighted_error)
        w_i = self.compute_weight(weighted_error)
        print('weight: ', w_i)
        self.estimator_weights.append(w_i)
        print('error:', weighted_error)
        new_alpha = self.compute_alpha(Y, Y_pred, w_i)
        # Normalized data weight
        self.alpha = new_alpha / np.sum(new_alpha)
        
    def compute_alpha(self, Y, Y_pred, w_i):
        n = Y.shape[0]
        Y_true = (Y_pred == Y).reshape((n, 1))
        exponential = (np.exp(np.where(Y_true, -1, 1) * w_i)).reshape((n, 1))
        
        return np.multiply(self.alpha, exponential)
    
    def compute_weight(self, error):
        return np.log((1 - error) / error) / 2
        
    def estimate_error(self, Y, Y_pred):
        return np.sum(np.multiply((Y != Y_pred), self.alpha))
    
    # Loading previously trained model
    def load_weights(self, summary):
        self.nn_layers = summary['nn_layers']
        self.estimator_dirs = summary['estimator_dir']
        self.estimator_weights = summary['estimator_weights']
        
    def predict(self, X):
        n = X.shape[0]
        pred_prob = np.zeros((n, 1))
        
        for i, dir in enumerate(self.estimator_dirs):
            cnn = CNN(self.nn_layers[i])
            model = cnn.build_model(X.shape[1])
            latest = tf.train.latest_checkpoint(os.path.join(dir, 'training_checkpoints'))
            model.load_weights(latest)

            pred_prob += (self.estimator_weights[i] * model.predict(X))
            
            del model
            self.clear_memory(cnn)
            
        pred_prob /= sum(self.estimator_weights)
        
        return pred_prob