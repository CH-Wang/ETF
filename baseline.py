import numpy as np 
import pandas as pd

## Persistence Model
class PersistenceModel():

    def fit(self, data, target):
        return True

    def predict(self, data, ndays=5):
    ## data: [[0,1,0,...], [1,0,1,...], ...]
        output = []
        for i in data:
            output.append([i[-1] for j in range(ndays)])
        return output
