import numpy as np

from ml_algo_script.supervised_learning.regression import Excutation

if __name__=="__main__":
    X=[[1,2,3],[4,5,6]]
    x_array=np.array(X)
    Y=[4,5]
    test_X=[[5,7,8],[9,8,1]]


    linerregression= Excutation(1000,3,0.001,X,Y,test_X)
    linerregression.modeul(x_array,Y,test_X)



