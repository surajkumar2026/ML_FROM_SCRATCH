import numpy as np
import sklearn 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    # Load the iris dataset
    data = sklearn.datasets.load_iris()

    x= data.data
    y= data.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train= np.array(x_train)

    from ml_algo_script.supervised_learning.logistic_regression import Logistic_regression
    from ml_algo_script.utils.matrix import accuracy,mse

    logistic_regression = Logistic_regression(no_itration=3000, learning_rate=0.0001)
    logistic_regression.initialize_parameters(x_train)
    logistic_regression.fit(x_train, y_train)
    y_pred = logistic_regression.predict(x_test)
    print("Predicted values:", y_pred)
    print("Actual values:", y_test)
    
    print("Accuracy:", accuracy(y_test, y_pred))
    print("MSE:", mse(y_test, y_pred))
if __name__ == "__main__":
    main()