
import numpy as np
from itertools import combinations_with_replacement


def polynomial_feature(X,degree):
    no_sample,no_feature= X.shape


    def index_combinations():
        combs = [combinations_with_replacement(range(no_feature), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    


    combination= index_combinations()
    no_output_feature=len(combination)
    X_new= np.empty((no_sample,no_output_feature))
    

    for i , index_com in enumerate(combination):
        X_new[:,i]= np.prod(X[:,index_com],axis=1)


    return X_new
