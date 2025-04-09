import numpy as np
import math


from ml_algo_script.utils.data_manipulation import sigmoid

class Logistic_regression():

    def __init__(self, no_itration, learning_rate):
        self.no_itration=no_itration
        self.learning_rate= learning_rate

    


    def initialize_parameters(self, X):
        print(X.shape)

        no_samples,no_feature=X.shape

        limit = 1/math.sqrt(no_feature)
        self.param= np.random.uniform(-limit,limit,(no_feature,)) ## uniform distribution kiya


    def fit(self, X_train,Y_train):


        for i in range (self.no_itration):
            Y_pred=X_train.dot(self.param)
            Y_pred_sigmoid= sigmoid(Y_pred)

            ## try to find the best apram to minimize the loss function
            #with help of gradient descent
            print("Y_pred_sigmoid",Y_pred_sigmoid.shape)
            print("Y_train",Y_train.shape)
            print("X_train",X_train.shape)
            print("self.param",self.param.shape)

            gradient= -(Y_train-Y_pred_sigmoid).dot(X_train)
            self.param -= self.learning_rate * gradient

    
    def predict(self, X_test):

        y_pred=X_test.dot(self.param)
        y_pred_sigmoid= sigmoid(y_pred)
        return np.round(y_pred_sigmoid)
        

