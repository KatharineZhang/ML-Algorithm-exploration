from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 6 * np.sin(np.pi * x) * np.cos(4 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    # multiply every value in x_i with x_j
    K = (np.multiply.outer(x_i, x_j) + 1)**d
    return K


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    matrix_subtr = np.subtract.outer(x_i, x_j)
    sqrd = -gamma*(matrix_subtr)**2
    K = np.exp(sqrd)
    return K


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """

    # FIND EQUATION IN NOTES:
    # Train is definted as:
        #
    reg = _lambda * np.identity(len(x))
    kernel = kernel_function(x, x, kernel_param)
    alpha_V = np.linalg.solve((kernel + reg), y)
    return alpha_V


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across all folds.
    """
    fold_size = len(x) // num_folds
    #calculate average loss
    loss = 0
    # use LOO method
    for i in range(num_folds):
        # get training set (all except i)
        up_to_i = i*fold_size
        after_i = (i+1)*fold_size
        x_train = np.concatenate((x[:up_to_i], x[after_i:]))
        y_train = np.concatenate((y[:up_to_i], y[after_i:]))
        # get validation sample i
        x_value = x[up_to_i:after_i] 
        y_value = y[up_to_i:after_i]
        # train
        alpha = train(x_train, y_train, kernel_function, kernel_param, _lambda)

        # calculate loss (RSS):
            # average of (y - K * alpha)^2
        kernel = kernel_function(x_value, x_train, kernel_param)
        loss += np.mean((y_value - kernel @ alpha)**2)

    # calculate avg loss
    avg = loss / num_folds
    return avg
        





@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    # use the Gird Search method
    # Remember: cross validation is used to determine the best regularization parameter lambda!

    # method: 
        # calculate number of possible lambdas using gamma equation given
        # loop over the lambdas
        # for every lambda, us cross validation to calculate loss
        # if lambda loss is < the best one so far, update
        # return best lambda

    
    lambdas_possible = []
    for i in np.linspace(-5, -1):
        # Calculate 10 raised to the power of each element and append
        lambdas_possible.append(10**i)
    
    best_loss = np.inf # set to +infinity bc we want to find the minimum loss
    best_lambda = 0

    # calculate gamma
    gamma = 1 / np.median(np.linalg.norm(x[:,None]-x, axis=-1)**2)
    
    # search for the best lambda
    for _lambda in lambdas_possible:
        loss = cross_validation(x, y, rbf_kernel, gamma, _lambda, num_folds)
        if loss < best_loss:
            best_loss = loss
            best_lambda = _lambda
    return best_lambda, gamma


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """
    # Use Grid Search method again
    lambdas_possible = []
    for i in np.linspace(-5, -1):
        # Calculate 10 raised to the power of each element and append
        lambdas_possible.append(10**i)

    min_loss = np.inf
    # find the best lambda and d value
        # reminder: d is the degree for poly reg
    for _lambda in lambdas_possible:
        for d in range(5, 26):
            loss = cross_validation(x, y, rbf_kernel, d, _lambda, num_folds)
            if loss < min_loss: # record lowest loss
                min_loss = loss
                best_lambda = _lambda
                best_d = d
    best_pair = (best_lambda, best_d)
    return best_pair

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    
    # Part A: Report the values of d, λ, and γ for both poly + rbf kernels
    lambda_rbf, gamma = rbf_param_search(x_30, y_30, len(x_30))
    lambda_poly, d = poly_param_search(x_30, y_30, len(x_30))
    print(f"RBF: lambda - {lambda_rbf}   gamma - {gamma}")
    print(f"POLY: lambda - {lambda_poly}   degree - {d}")

    # Part B: plot the original data + the true f(x) and f(x)_learned

    # expected
    fx_true = f_true(np.linspace(0, 1, num=100))

    # prediction = Kernel * alpha

    alpha_rbf = train(x_30, y_30, rbf_kernel, gamma, lambda_rbf)
    alpha_poly = train(x_30, y_30, poly_kernel, d, lambda_poly)

    kernel_rbf = rbf_kernel(np.linspace(0, 1, num=100), x_30, gamma)
    kernel_poly = poly_kernel(np.linspace(0, 1, num=100), x_30, d)

    pred_rbf = np.dot(kernel_rbf, alpha_rbf)
    pred_poly = np.dot(kernel_poly, alpha_poly)

    # generate fine grid which is defined as np.linspace(0, 1, num=100)
    grid = np.linspace(0, 1, num=100)

    # RBF plot
    plt.figure(1)
    plt.scatter(x_30, y_30, label="Data")
    plt.plot(grid, fx_true, label="True function")
    plt.plot(grid, pred_rbf, label="RBF")
    plt.ylim(-6, 6)
    plt.legend()
    plt.show()

    # Poly plot
    plt.figure(2)
    plt.plot(grid, fx_true, label="True function")
    plt.plot(grid, pred_poly, label="Poly")
    plt.ylim(-6, 6)
    plt.legend()    
    plt.show()


if __name__ == "__main__":
    main()
