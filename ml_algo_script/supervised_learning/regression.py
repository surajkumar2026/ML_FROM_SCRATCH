import numpy as np

from ml_algo_script.utils.data_manipulation import polynomial_feature
from ml_algo_script.utils.matrix import mse

class Regression:
    def __init__(self, no_itration,no_features,learning_rate):
        self.no_itration = no_itration
        self.no_features = no_features
        self.learning_rate= learning_rate
        
    def initialize_weights(self, no_features):
         """initialize the weights for the regression model"""
         # The limit isliye set kiya hai ki weight ki value bahut zyada nahi ho jaye, to prevnet from overfitting
      
         limit= 1/(no_features+1)**0.5
         self.w = np.random.uniform(-limit, limit, (no_features,))## uniform distribution kiya 

    def fit(self,X,Y):
    
        X=np.insert(X,0,1,axis=1) # yeha intercept add kiya gaya hai.
        self.initialize_weights(X.shape[1])
        self.error = [] ## for containing the error of each iteration
        ## now do gradient descent
        ## gradient desent kar ke weight ko update karna hai

        for i in range(self.no_itration):
            Y_pred= X.dot(self.w)
            ## use MSE as the cost function 
            error= mse(Y,Y_pred)
            # error= np.mean((Y - Y_pred) ** 2)
            self.error.append(error)
            grad_w= -(Y-Y_pred).dot(X)
            #update the weights
            self.w -= self.learning_rate*grad_w

    def prediction(self, X):
        X=np.insert(X,0,1,axis=1)
        Y_pred= X.dot(self.w)
        return Y_pred
    


class Linear_regression(Regression):
    def __init__(self, no_itration, no_features, learning_rate):
        super(Linear_regression,self).__init__(no_itration= no_itration, no_features= no_features,learning_rate=learning_rate)

    def fit_model(self,X,Y):
         super(Linear_regression,self).fit(X,Y)

    def predict(self,X):
         return super(Linear_regression,self).prediction(X)
    



class polynomial_regression(Regression):
    def __init__(self,degree, no_itration, no_features, learning_rate):
        self.degree= degree
        super(polynomial_regression,self).__init__(no_itration, no_features, learning_rate)


    def fit_model(self, X, Y):
        X= polynomial_feature(X,degree=self.degree)
        return super(polynomial_regression,self).fit(X, Y)


    def pridict(self,X):
        X=polynomial_feature(X,degree=self.degree)
        return super(polynomial_regression,self).prediction(X)



class module_regression():

    def __init__(self,no_itration,no_feature,learning_rate):
        self.no_itration=no_itration
        self.no_feature=no_feature
        self.learning_rate=learning_rate
        
       
    
    def linear_modeul(self,X,Y,test_X):
        Linearregression = Linear_regression(self.no_itration,self.no_feature,self.learning_rate)
        fit_data=Linearregression.fit_model(X,Y)
        print(Linearregression.predict(test_X))


    def poly_regression_model( self,degree, X,Y,test_X):
        polynomialregression= polynomial_regression(degree,self.no_itration,self.no_feature,self.learning_rate)
        fit_data= polynomialregression.fit_model(X,Y)
        print(f"polynomialregression prediction is {polynomialregression.pridict(test_X)}")
        pass





 


    










        


       



    


    






        