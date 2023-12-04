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


class AdaBoost:
    def __init__(self, nn=[]):
        # List of number of layers in base estimators shallow CNN
        self.nn = []
        self.alphas = [] # Data weight
        self.estimator_weights = [] # Estimator weight, w
        self.estimators = []
        
        
        
    def predict_prob(self, X):
        prob = sum(estimator.predict_prob(X) * w for estimator, w in zip(self.estimators, self.estimator_weights))
        prob /= sum(self.estimator_weights)
        
        return proba