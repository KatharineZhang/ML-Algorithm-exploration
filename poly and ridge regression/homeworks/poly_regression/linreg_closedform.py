"""
    Sample implementation of linear regression using direct computation of the solution
    AUTHOR Eric Eaton
"""

import numpy as np

"""
-----------------------------------------------------------------
 Class LinearRegression - Closed Form Implementation
-----------------------------------------------------------------
"""


class LinearRegressionClosedForm:
    def __init__(self, reg_lambda=1e-8):
        """
        Constructor
        """
        self.regLambda = reg_lambda
        self.theta = None

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-d array
                y is an n-by-1 array
            Returns:
                No return value
        """
        n = len(X) # a 2D array??? Does it get the # of columns or the # of rows??

        # add 1s column
        # np.ones creates a new oarray of given shape and type, filled with 1's
        # np.c_[] stacks the 2 1D arrays into a 2D array named X_
            # stacks a row-array of nx1 with our X matrix of dimensions nxd where d = 8.
        X_ = np.c_[np.ones([n, 1]), X]

        # array.shape will give us the dimensions of our 2D array in the form (rows, cols)
        n, d = X_.shape
        # remove 1 for the extra column of ones we added to get the original num features
        d = d - 1

        # construct reg matrix
        # np.eye(N, M) returns an identity matrix
            # N = number of rows, which in this case is d + 1
            # M = number of columns, which, if not specified, is same as N (d+1)
        reg_matrix = self.regLambda * np.eye(d + 1)
        reg_matrix[0, 0] = 0 #Note that we do not penalize the offset term w0, 
                             # since that only affects the global mean of the output, 
                             #and does not contribute to overfitting

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.solve(X_.T @ X_ + reg_matrix, X_.T @ y)

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]

        # predict
        return X_.dot(self.theta)


"""
-----------------------------------------------------------------
 End of Class LinearRegression - Closed Form Implementation
-----------------------------------------------------------------
"""
