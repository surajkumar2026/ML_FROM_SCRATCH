import numpy as np
import math


from ml_algo_script.utils.data_manipulation import sigmoid

class Logistic_regression():

    def __init__(self, no_itration, learning_rate):
        self.no_itration=no_itration
        self.learning_rate= learning_rate

    


    def initialize_parameters(self, X):

        no_samples,no_feature=X.shape

        limit = 1/math.sqrt(no_feature)
        self.param= np.random.uniform(-limit,limit,no_feature)


    def fit(self, X_train,Y_train):


        for i in range (self.no_itration):
            Y_pred=X_train.dot(self.param)
            Y_pred_sigmoid= sigmoid(Y_pred)

            ## try to find the best apram to minimize the loss function
            #with help of gradient descent

            gradient= -(Y_train-Y_pred_sigmoid).dot(self.param)
            self.param -= self.learning_rate * gradient

    
    def predict(self, X_test):

        y_pred=X_test.dot(self.param)
        y_pred_sigmoid= sigmoid(y_pred)
        return y_pred_sigmoid
        

