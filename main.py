import numpy as np
import pandas as pd

from ml_algo_script.supervised_learning.regression import module_regression

from ml_algo_script.model_test.Logistic_Regression import main

from ml_algo_script.model_test.SVM import svm





if __name__=="__main__":
    # X=[[1,2,3],[4,5,6]]
    # x_array=np.array(X)
    # Y=[4,5]
    # test_X=np.array([[5,7,8],[9,8,1]])
    
    # regression= module_regression(1000,3,0.0001)
    # regression.linear_modeul(x_array,Y,test_X)
    # regression.poly_regression_model(degree=2, X=x_array, Y=Y,  test_X=test_X)
    logistic_test=main()
    SVM = svm()
    






