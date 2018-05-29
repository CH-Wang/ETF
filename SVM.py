import numpy as np 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import pandas as pd
from sklearn.externals import joblib

class SVM():
    def __init__(self):
        self.model = MultiOutputRegressor(SVR(kernel='rbf', C=1e3, gamma=0.1))

    def fit(self, train_input, train_target):
        self.model.fit(train_input,train_target)

    def predict(self, test_input):
        return self.model.predict(test_input)

    def save(self):
        joblib.dump(self.model, 'model/SVM.pkl')

    def load(self):
        self.model = joblib.load('model/SVM.pkl')

