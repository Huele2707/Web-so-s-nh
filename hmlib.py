#import libary
import pickle
import numpy as np
import pandas as pd




#Package model
class Iris_Model:
    target_name = ['setosa', 'versicolor', 'virginica']
    def __init__(self):
        self.model = pickle.load(open('iris.pickle','rb'))
    def processing(self, x):
        x = np.array(x).reshape(-1,4)# đưa về mảng 2 chiều dạng -1,4 
        return x
    def predict(self,x):
        result = np.array([])
        x = self.processing(x)
        y = self.model.predict(x)
        for i in y:
            result = np.append(result,self.target_name[i])
        return result
    def predict_probability(self,x):
        x = self.processing(x)
        y = self.model.predict_proba(x)
        result = pd.DataFrame(y,columns=self.target_name)
        return result
        