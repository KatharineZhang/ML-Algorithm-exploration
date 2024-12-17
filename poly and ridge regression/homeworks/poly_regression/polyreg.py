"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        self.variance: np.ndarray
        self.mean: np.ndarray

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        # in the demo, they did this: 
        # X = np.vstack([np.ones(len(x)),x,x**2,x**3,x**4]).T for degree = 4
        # we don't know what degree is, so we will have to iterate (i think)
        # Q: why do we put it into a matrix/why do we use equation
        poly_X = np.zeros((len(X), degree))
        for i in range(1, degree + 1):
            poly_X[:, i - 1] = X[:, 0]**i
        return poly_X


    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        reg_X = self.polyfeatures(X, self.degree)

        # standardize: (x - mu) / standardization
        self.mean = np.mean(reg_X, axis = 0)
        self.variance = np.std(reg_X, axis = 0)
        stnd_X = (reg_X - self.mean) / self.variance

        #create an nx1 matrix and add it to stnd_X for bias
        # Q: do we really need this?
        stnd_X = np.hstack((np.ones((len(stnd_X), 1)), stnd_X))

        # set up regularization matrix
        reg_matrix = self.reg_lambda * np.eye(self.degree + 1)
        reg_matrix[0, 0] = 0 #Note that we do not penalize the offset term w0, 
                             # since that only affects the global mean of the output, 
                             #and does not contribute to overfitting

        # according to the texbook, we have the equations:
        # y_hat = Xw = X(X^T X)^{-1}X^T y
        # our weight = (X^T X + regLambda)^{-1}X^T y
        X_TX = np.matmul(np.transpose(stnd_X), stnd_X)
        X_Ty = np.matmul(np.transpose(stnd_X), y)
        self.weight = np.matmul(np.linalg.inv(X_TX + reg_matrix), X_Ty)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        # in lecutre slides, prediction y_new = x^T_i,new * weight + offset = Xw + offset
        # there is also the y_hat = Xw equation somewhere for d-dimensional?
        # offset is b = (1/n) sum_{1}^{n} y_i
        reg_X = self.polyfeatures(X, self.degree)

        # standardize: (x - mu) / standardization
        stnd_X = (reg_X - self.mean) / self.variance

        #create an nx1 matrix and add it to stnd_X for bias
        stnd_X = np.hstack((np.ones((len(stnd_X), 1)), stnd_X))

        # Chage to: y = Xw
        return np.matmul(stnd_X, self.weight)



@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    MSE = np.mean((a-b)**2)
    return MSE



@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    # 1) train Xtrain and Ytrain (fit, then predict) using [: i] for both
    # 2) errorTrain at i-1 = MSE of Xtrain and Ytrain

    # 3) train Xtest and Ytest (fit, then predict) using [: i] for both
    # 4) errorTest at i-1 = MSE of Xtest and Ytest

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    polyreg = PolynomialRegression(degree, reg_lambda)

    for i in range(1, n):
        # find the prediction for training data
        polyreg.fit(Xtrain[0:(i + 1)], Ytrain[0:(i + 1)])
        pred_Xtrain = polyreg.predict(Xtrain[0:(i + 1)])
        # calculate errorTrain
        errorTrain[i] = mean_squared_error(pred_Xtrain, Ytrain[0:(i + 1)])
        
        # WHY DO WE NOT GO FROM O - i + 1 ????
        # find the prediction for the test data
        pred_Xtest = polyreg.predict(Xtest)
        # calculate errorTest
        errorTest[i] = mean_squared_error(pred_Xtest, Ytest)

        # so we have to train from 1 - 2, 1-3 ,1 - i+1 data points...
    return errorTrain, errorTest
