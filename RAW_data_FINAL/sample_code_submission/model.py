'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling2D
from keras.layers import Activation, Reshape
from keras import backend as K
import matplotlib.pyplot as plt

''' On définit un réseau de neuronne avec le module Keras composé de 4 couches dont la dernière qui retourne la classe prédite'''
def ourNetwork():
    model = Sequential()
    model.add(Dense(128, input_dim=200, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(2, kernel_initializer='uniform', activation='elu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

class model (BaseEstimator):
    ''' On initialise notre neuronne'''
    def __init__(self):
        self.model = ourNetwork()

    ''' On effectue la phase d'apprentissage sur 15 époques. Cette méthode permet aussi de tracer
	un courbe sur la valeur Accurancy et l'autre sur le Loss en fonction des époques'''
    def fit(self, X, y):
        history = self.model.fit(X, y,epochs=30, validation_split=0.2,batch_size=25)
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy (Multi Layer Perceptron)')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss (Multi Layer Perceptron)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    ''' On effectue les prédictions avec des probabilités pour obtenir une courbe ROC plus juste'''
    def predict(self, X):
        y = self.model.predict_proba(X)
        return y

    def save(self, path="./"):
        pickle.dump(self.model, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
