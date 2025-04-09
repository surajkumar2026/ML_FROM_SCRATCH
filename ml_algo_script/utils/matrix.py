import numpy as np






def mse(y_true, y_pred):
    error= np.mean((y_true - y_pred) ** 2)
    return error

def accuracy (y_true,y_pred):
    return np.sum(y_true==y_pred,axis=0)/len(y_true)



