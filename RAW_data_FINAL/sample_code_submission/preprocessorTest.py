from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from preprocessor import Preprocessor
import numpy as np
import unittest
np.seterr(divide='ignore', invalid='ignore')

#Unit tests for the 'Preprocessor' class.
class TestPreprocessor(unittest.TestCase):
    
    #Test the 'standardizing' function.
    def testStandardizing(self):
        self.prep = Preprocessor()
        self.scaler = StandardScaler()
        self.X_unitS = np.empty([10, 10])
        self.X_unitS2 = np.empty([10, 10])
        for i in range(10):
            self.X_unitS[i] = range(10)
            self.X_unitS2[i] = range(10)
        self.X_unitS = self.prep.standardizing(self.X_unitS)
        self.X_unitS2 = self.scaler.fit_transform(self.X_unitS2)
        for i in range(10):
            for j in range(10):
                self.assertEqual(self.X_unitS[i][j], self.X_unitS2[i][j])
                
    #Test the 'tranfsorm' function.            
    def testTransform(self):
        self.prep = Preprocessor()
        self.pca = IncrementalPCA(n_components=200)
        self.X_unitP = np.empty([300, 500])
        self.X_unitP2 = np.empty([300, 500])
        for i in range(300):
            self.X_unitP[i] = range(500)
            self.X_unitP2[i] = range(500)
        self.prep.partial_fit(self.X_unitP)
        self.X_unitP = self.prep.transform(self.X_unitP)
        self.X_unitP2 = self.pca.fit_transform(self.X_unitP2)
        for i in range(300):
            for j in range(200):
                self.assertEqual(self.X_unitP[i][j], self.X_unitP2[i][j])
        
        
if __name__=='__main__':
    unittest.main(argv =[''], verbosity=2, exit=False)
