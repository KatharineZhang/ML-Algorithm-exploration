from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, step_size: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        step_size (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """

    # step is a helper method that train() uses
    # this covers the for loop part! NOt the while part. The while-convergence part is within train
    bias_new = bias - 2 * step_size * np.sum(X @ weight + bias - y)
    t = np.matmul(X, weight) + bias - y #500 x 500 when it should be 500 x 1
    v = np.matmul(np.transpose(X), t)
    
    #print(y.shape)
    #print(X.shape)
    #print(weight.shape)
    #print(s.shape)
    #print(t.shape)
    #print(v.shape)
    w_new = weight - 2 * step_size * v

    w_return = np.where(w_new < -2 * step_size * _lambda, w_new + 2 * step_size * _lambda, 
                        np.where(w_new > 2 * step_size * _lambda, w_new - 2 * step_size * _lambda, 0))
    
    return w_return, bias_new
     


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """

    # use SSE function:
    #               argmin sum_{i=1}^{n}(y_i - x_i.transpose * weight)^2 + lambda||weight||
    # remember: equation for 1-norm is sum_{i=1}^{n} |x| = ||x||_1

    # I'm missing the bias variable...

    # add bias within (y_i - x_i.transpose * weight)^2, ASK WHY ON ED
    loss = np.linalg.norm(y - (X @ weight + bias))**2 + _lambda * np.linalg.norm(weight, ord=1)
    return loss
    



@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: float = None

    #while not converge
        # save b and w (these are our old values in which we would use in c_c())
        # call step
        # b = b' and w = w'
    bias = start_bias
    weight = np.copy(start_weight)
    is_converged = False
    while (not is_converged):
        old_b = bias
        old_w = np.copy(weight)

        weight, bias = step(X, y, weight, bias, _lambda, eta)
        is_converged = convergence_criterion(weight, old_w, bias, old_b, convergence_delta)
    
    return weight, bias


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    # w_old - weight (which is new) and take that 
    # absolute max change and compare with convergence_delta
    w_change = np.abs(np.max(old_w - weight))
    b_change = np.abs(np.max(old_b - bias))

    return (w_change <= convergence_delta) and (b_change <= convergence_delta)
    


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """

    # generate a data set with:
    # n=500, d = 1000, k=100, and variance = 1

    # Set random seed for reproducibility
    np.random.seed(500)

    # Define the number of samples and features
    n = 500
    d = 1000
    k = 100

    # Generate synthetic data
    X = np.random.normal(loc=0, scale=1, size=(n, d))

    # Generate synthetic y's where y = x^T w + e
    w = np.concatenate((np.arange(1, k+1) / k, np.zeros(d - k)))
    noise = np.random.normal(0, 1, n)
    y = np.matmul(X, w) + noise

    # standardize X
    stnd_X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)  
   
    
    # lambda_max = 
        # max_{k = 1,...,d} 2 * |sum_{i=1}^{n} x_{i,k} * (y_i - [(sum_{j=1}^{n} y_j) / n)]|
        # sum_{i=1}^{n} x_{i,k} = for every feature in every sample in our X matrix (aka X^T)
        # (sum_{j=1}^{n} y_j) / n) = the mean of our y vector
    lambda_max = np.max(2 * np.abs(np.transpose(stnd_X) @ (y - np.mean(y))))
    lambda_ = lambda_max 
    
    # a loop that decreases lambda from max to 0, dividing 2 each time
    # _lambda = lambda_max. Use _lambda instead of lambda_max, _lambda = _lambda/2
    # max number of features is 1000. So to check if all features are chosen, check how many weights
    # are NOT 0, and once that reaches a certain threshold, stop the loop
    lambda_list = []
    lambda_list.append(lambda_)
    features_list = []
    weight, bias = train(stnd_X, y, lambda_)
    features_list.append(np.count_nonzero(weight))

    FDR = []
    TPR = []
    features_selected = 0
    while lambda_ > 0.01:
        lambda_ = lambda_ / 2 #update lambda

        weight, bias = train(stnd_X, y, lambda_)
        features_list.append(features_selected)

        # a) make a graph of lambda vs features selected
        features_selected = np.sum(1 - np.isclose(weight, 0))
        lambda_list.append(lambda_)
        

        # b) make a graph of FDR (# of  incorrect nonzeros in w/total numer of nonzeros in w) VS
        #   TPR (# of correct nonzeros in w/k)
                # incorrect nonzero is defined by predicted w != 0 while actual w = 0
        if features_selected != 0:
            FDR.append(np.count_nonzero(weight[k+1:d]) / features_selected)
            TPR.append((features_selected - np.count_nonzero(weight[k:])) / k)
        else:
            FDR.append(0)
            TPR.append(0)
    
        
        
    # part a graph
    plt.figure(1)
    plt.plot(lambda_list, features_list)
    plt.xscale("log")
    plt.show()

    # part b graph
    plt.figure(2)
    plt.plot(FDR, TPR)
    plt.show()

if __name__ == "__main__":
    main()
