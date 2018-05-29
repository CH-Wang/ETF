import numpy as np 
import pandas as pd


## Persistence Model
class PersistenceModel():

    def fit(self, data, target):
        return True

    def predict(self, data):
        output = []
        for i in data:
            output.append([i[-1] for j in range(5)])
        return output
