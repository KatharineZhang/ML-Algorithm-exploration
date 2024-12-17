if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    # set model to evaluation mode
    model.eval()

    with torch.no_grad():
        correct_pred = 0
        total_samples = 0
        for _input, expected in dataloader:
            outputs = model(_input)
            predictions = torch.argmax(outputs, dim=1)
            exp = torch.argmax(expected, dim=1)
            correct_pred += (predictions == exp).sum().item()
            total_samples += outputs.shape[0]
    accuracy = correct_pred / total_samples
    return accuracy


@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, 
    and return a dictionary containing training history of models, 
    as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    history = {}

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
    # Note: 
    # nn.Sequential() is a container that allows you to chain 
    # together multiple neural network layers in a sequential manner.
    # It provides a convenient way to organize and construct neural 
    # network architectures

    # Linear Regression Model - use the LinearLayer to represent this
    layers_1 = [LinearLayer(2, 2, generator=RNG)]
    linear_model = nn.Sequential(*layers_1)
    linear_opt = SGDOptimizer(linear_model.parameters(), lr=1e-3)
    linear_history = train(train_loader, linear_model, MSELossLayer(), linear_opt, val_loader, epochs=50)
    history["linear"] = {"train": linear_history["train"], "val": linear_history["val"], "model": linear_model}

    # Network with one hidden layer of size 2 
    # and sigmoid activation function after the hidden layer
    layers_2 = [LinearLayer(2, 2, generator=RNG), SigmoidLayer(), 
               LinearLayer(2, 2, generator=RNG),]
    sigmoid_model = nn.Sequential(*layers_2)
    sigmoid_opt = SGDOptimizer(sigmoid_model.parameters(), lr=1e-3)
    sigmoid_history = train(train_loader, sigmoid_model, MSELossLayer(), 
                            sigmoid_opt, val_loader, epochs=50)
    history["NNHiddenSigmoid"] = {"train": sigmoid_history["train"], 
                                  "val": sigmoid_history["val"], "model": sigmoid_model}

    # Network with one hidden layer of size 2 
    # and ReLU activation function after the hidden layer
    layers_3 = [LinearLayer(2, 2, generator=RNG), ReLULayer(), 
               LinearLayer(2, 2, generator=RNG),]
    relu_model = nn.Sequential(*layers_3)
    relu_opt = SGDOptimizer(relu_model.parameters(), lr=1e-3)
    relu_history = train(train_loader, relu_model, MSELossLayer(), 
                         relu_opt, val_loader, epochs=50)
    history["NNHiddenReLU"] = {"train": relu_history["train"], 
                               "val": relu_history["val"], "model": relu_model}

    # Network with two hidden layers (each with size 2)
    #        and Sigmoid, ReLU activation function after corresponding hidden layers
    layers_4 = [LinearLayer(2, 2, generator=RNG), SigmoidLayer(), 
               LinearLayer(2, 2, generator=RNG), ReLULayer(), 
               LinearLayer(2, 2, generator=RNG),]
    sig_relu_model = nn.Sequential(*layers_4)
    sig_relu_opt = SGDOptimizer(sig_relu_model.parameters(), lr=1e-3)
    sig_relu_history = train(train_loader, sig_relu_model, MSELossLayer(), 
                             sig_relu_opt, val_loader, epochs=50)
    history["NNHiddenSigmoidReLU"] = {"train": sig_relu_history["train"], 
                                      "val": sig_relu_history["val"], "model": sig_relu_model}

    # Network with two hidden layers (each with size 2)
    #        and ReLU, Sigmoid activation function after corresponding hidden layers
    layers_5 = [LinearLayer(2, 2, generator=RNG), ReLULayer(), 
               LinearLayer(2, 2, generator=RNG), SigmoidLayer(), 
               LinearLayer(2, 2, generator=RNG),]
    relu_sig_model = nn.Sequential(*layers_5)
    relu_sig_opt = SGDOptimizer(relu_sig_model.parameters(), lr=1e-3)
    relu_sig_history = train(train_loader, relu_sig_model, MSELossLayer(), 
                             relu_sig_opt, val_loader, epochs=50)
    history["NNHiddenReLUSigmoid"] = {"train": relu_sig_history["train"], 
                                      "val": relu_sig_history["val"], "model": relu_sig_model}
    return history


@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
    )
    # 1. Call mse_parameter_search routine and get dictionary for 
    # each model architecture/configuration.
    mse_configs = mse_parameter_search(dataset_train, dataset_val)
    
    plt.figure(1)
    best_model = None
    best_val_loss = np.inf
    best_model_name = None
    for model_name, model_params in mse_configs.items():
        val_error = model_params.get("val")
        if min(val_error) < best_val_loss:
            best_val_loss = min(val_error)
            best_model = model_params.get("model")
            best_model_name = model_name
        plt.plot(model_params["train"], label=f"{model_name}" + " train")
        plt.plot(model_params["val"], label=f"{model_name}" + " val")
    plt.title("Train and Validation loss for different models using MSE")
    plt.legend()
    plt.show()

    plt.figure(2)
    plot_model_guesses(DataLoader(dataset_test), best_model, "best model NN Predictions using MSE loss")
    plt.show()

    print("Best Model Name: ", best_model_name)
    print("Best Model Validation Loss: ", best_val_loss)
    print("Accuracy score: ", accuracy_score(best_model, DataLoader(dataset_test, batch_size=32, shuffle=False)))



def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
