from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
import sys

class Preprocessor:

    def __init__(self):
        self.pca = IncrementalPCA(n_components = 200)

    """

    Standardize the data with the module sklearn.preprocessing.StandardScaler

    Parameters
    ----------
    X (2D array): The data to standardize

    Returns
    -------
    2D array: The data standardized

    """
    def standardizing(self, X, y=None):
        scaler = StandardScaler()
        new_X = np.asfarray(X, dtype='float')
        return scaler.fit_transform(new_X)


    """Apply the 'partial_fit' function of the module sklearn.decomposition.IncrementalPCA on the daya"""
    def partial_fit(self, X, y=None):
        return self.pca.partial_fit(X, y)


    """

    Transform the data with the module sklearn.decomposition.IncrementalPCA

    Parameters
    ----------
    X (2D array): the data to transform

    Returns
    -------
    2D array: the data transformed

    """
    def transform(self, X, y=None):
        print("Transforming \n")
        X_shape = X.shape[0]
        new_X = np.empty([X_shape, 200])
        i = 0
        transformed = False
        while (transformed == False):
            if (X_shape - i <= 50):
                partial_X = X[i:]
                new_X[i:] = self.pca.transform(partial_X)
                transformed = True
            else:
                partial_X = X[i:i+50]
                new_X[i:i+50] = self.pca.transform(partial_X)
                i += 50
                print(str((i*100)//X_shape)+" %", end='\r')
        print("100 %", end='\r')
        return new_X
