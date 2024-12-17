if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train 

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    # split into x/y train and x/y test
    # train.to_numpy
    x_train = df_train.drop(columns=["ViolentCrimesPerPop"]) # drop ViolentCrimesPerPop
    y_train = df_train["ViolentCrimesPerPop"]

    # our test case is the ViolentCrimesPerPop data set
    x_test = df_test.drop("ViolentCrimesPerPop", axis = 1) # drop all columns except ViolentCrimesPerPop
    y_test = df_test["ViolentCrimesPerPop"]
    

    # create lambda
        # lambda_max = 
        # max_{k = 1,...,d} 2 * |sum_{i=1}^{n} x_{i,k} * (y_i - [(sum_{j=1}^{n} y_j) / n)]|
    lambda_ = np.max(2 * np.abs(np.transpose(x_train) @ (y_train - np.mean(y_train))))

    lambda_list = []
    features_list = []

    agePct12t29= []
    pctWSocSec = []
    pctUrban = []
    agePct65up = []
    householdsize = []

    train_errors = []
    test_errors = []

    old_w = np.zeros(x_train.shape[1])
    old_b = 0

    # create a while loop to test for every single lambda
        #run lasso stuff (call your own train functions and get test + train error)
    d = x_train.shape[1]
    threshold = d * 0.01
    # append results to plot
    features_selected = 0
    while lambda_ > 0.01:

        weight, bias = train(x_train, y_train, lambda_, eta=0.00001, convergence_delta=1e-2,
                            start_weight=old_w, start_bias=old_b)
        old_w = weight
        old_b = bias

        # (Calculate MSE for both train and test)
            # MSE = np.mean((a-b)**2)s
        train_prediction = x_train @ weight + bias
        error_train = np.mean((train_prediction - y_train) ** 2)

        test_prediction = x_test @ weight + bias
        error_test = np.mean((test_prediction - y_test) ** 2)

        # part c: Plot the number of nonzero weights of each solution as a function of λ
        features_selected = np.count_nonzero(weight)
        lambda_list.append(lambda_)
        features_list.append(features_selected)

        # part d: Plot the regularization paths (in one plot) for the coefficients for input variables agePct12t29,
                # pctWSocSec, pctUrban, agePct65up, and householdsize.
        agePct12t29.append(weight[x_train.columns.get_loc("agePct12t29")])
        pctWSocSec.append(weight[x_train.columns.get_loc("pctWSocSec")])
        pctUrban.append(weight[x_train.columns.get_loc("pctUrban")])
        agePct65up.append(weight[x_train.columns.get_loc("agePct65up")])
        householdsize.append(weight[x_train.columns.get_loc("householdsize")])

        # part e: On one plot, plot the mean squared error on the training and test data as a function of λ.
        train_errors.append(error_train)
        test_errors.append(error_test)

        lambda_ = lambda_ / 2

    # 6c
    plt.title("lambda VS # of selected features") 
    plt.xscale("log")
    plt.plot(lambda_list, features_list)
    plt.show()

    #6d
    plt.title("Reg paths for coefficients")
    plt.xscale("log")
    plt.plot(lambda_list, agePct12t29, label = "agePct12t29")
    plt.plot(lambda_list, pctWSocSec, label = "pctWSocSec")
    plt.plot(lambda_list, pctUrban, label = "pctUrban")
    plt.plot(lambda_list, agePct65up, label = "agePct65up")
    plt.plot(lambda_list, householdsize, label = "householdsize")
    plt.legend()
    plt.show()

    #6e
    plt.title("lambda VS Train/Test error")
    plt.xscale("log")  
    plt.plot(lambda_list, train_errors, label = "Train Error")
    plt.plot(lambda_list, test_errors, label = "Test Error")
    plt.legend()
    plt.show()

    # 6f: plot all features
    weight, bias = train(x_train, y_train, 30, eta=0.00001, convergence_delta=1e-4,
                            start_weight=old_w, start_bias=old_b)
    max = np.argmax(weight)
    min = np.argmin(weight)
    feature_max = df_train.columns[max]
    feature_min = df_train.columns[min]
    print("The max feature " + feature_max)
    print("The min feature " + feature_min)

if __name__ == "__main__":
    main()
