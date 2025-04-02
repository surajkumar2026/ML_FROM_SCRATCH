import numpy as np






def mse(y_true, y_pred):
    error= np.mean((y_true - y_pred) ** 2)
    return error



