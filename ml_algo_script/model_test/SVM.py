from __future__ import division, print_function
import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split
from ml_algo_script.supervised_learning.support_vector_machine import SupportVectorMachine
from ml_algo_script.utils.matrix import accuracy,mse
from  ml_algo_script.utils.kernels import polynomial_kernel,rbf_kernel,linear_kernel


def svm():
    data = datasets.load_iris()
    X = data.data[data.target != 0]
    y = data.target[data.target != 0]
    y[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = SupportVectorMachine(kernal=polynomial_kernel, power=4,gamma=None, coef=1,c=1)
    clf.fit(X_train, y_train)
    y_pred = clf.prediction(X_test)

    accuracy_score = accuracy(y_test, y_pred)

    print ("Accuracy:", accuracy_score)




if __name__ == "__main__":
    svm()