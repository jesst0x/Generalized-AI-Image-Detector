import argparse
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import util


def build_model(initial_learning_rate=1e-5, freeze_layer=45, decay_steps=100, decay_rate=1):
    basemodel = ResNet50(include_top=False, weights='imagenet', pooling='max')
    
    x = basemodel.output
    # Add fully connected layer
    x = layers.Dense(1024, activation='relu')(x)
    # Logistic output layer
    predictions = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=basemodel.input, outputs=predictions)
    
    # Freeze first n layers of pre-trained RESNET50
    for layer in basemodel.layers[:freeze_layer]:
        layer.trainable = False
    for layer in basemodel.layers[freeze_layer:]:
        layer.trainable = True
    
    learning_rate = initial_learning_rate
    if decay_rate != 1:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,decay_steps=decay_steps , decay_rate=decay_rate, staircase=True)
    
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model

def train(X_train, Y_train, X_eval, Y_eval, num_epoch, batch_size, learning_rate, decay_rate, decay_epoch, freeze_layer, logging_dir):
    
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
    
    decay_steps = X_train.shape[0] // batch_size * decay_epoch
    
    # Fit ResNet50 model with transfer learning
    print('******Training model*****')
    model = build_model(learning_rate, freeze_layer, decay_steps, decay_rate)
    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(X_train, Y_train, epochs=num_epoch, batch_size=batch_size, callbacks =[cp_callback],validation_data=(X_eval, Y_eval), shuffle=True)   
    json.dump(history.history, open(os.path.join(logging_dir, 'history.json'), 'w'))
    print('Training completed!')
    print('Training result', history.history)
    
    # Plot loss and accuracy graph
    util.plot_history(history.history, logging_dir)

    return model