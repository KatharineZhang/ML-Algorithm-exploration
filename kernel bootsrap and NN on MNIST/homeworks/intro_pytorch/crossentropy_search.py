if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
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
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

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
    linear_history = train(train_loader, linear_model, CrossEntropyLossLayer(),
                            linear_opt, val_loader, epochs=50)
    history["linear"] = {"train": linear_history["train"], "val": linear_history["val"],
                          "model": linear_model}

    # Network with one hidden layer of size 2 
    # and sigmoid activation function after the hidden layer
    layers_2 = [LinearLayer(2, 2, generator=RNG), SigmoidLayer(), 
               LinearLayer(2, 2, generator=RNG),]
    sigmoid_model = nn.Sequential(*layers_2)
    sigmoid_opt = SGDOptimizer(sigmoid_model.parameters(), lr=1e-3)
    sigmoid_history = train(train_loader, sigmoid_model, CrossEntropyLossLayer(), 
                            sigmoid_opt, val_loader, epochs=50)
    history["NNHiddenSigmoid"] = {"train": sigmoid_history["train"], 
                                  "val": sigmoid_history["val"], "model": sigmoid_model}

    # Network with one hidden layer of size 2 
    # and ReLU activation function after the hidden layer
    layers_3 = [LinearLayer(2, 2, generator=RNG), ReLULayer(), 
               LinearLayer(2, 2, generator=RNG),]
    relu_model = nn.Sequential(*layers_3)
    relu_opt = SGDOptimizer(relu_model.parameters(), lr=1e-3)
    relu_history = train(train_loader, relu_model, CrossEntropyLossLayer(), 
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
    sig_relu_history = train(train_loader, sig_relu_model, CrossEntropyLossLayer(), 
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
    relu_sig_history = train(train_loader, relu_sig_model, CrossEntropyLossLayer(), 
                             relu_sig_opt, val_loader, epochs=50)
    history["NNHiddenReLUSigmoid"] = {"train": relu_sig_history["train"], 
                                      "val": relu_sig_history["val"], "model": relu_sig_model}
    return history


@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """

    # set model to evaluation mode
    model.eval()
    correct_pred = 0
    total_samples = 0
    with torch.no_grad():
        for _input, expected in dataloader:
            outputs = model(_input)
            predictions = torch.max(outputs, dim=1)
            correct_pred += (predictions[1] == expected).sum().item()
            total_samples += outputs.shape[0]
    accuracy = correct_pred / total_samples
    return accuracy


@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    # 1. Call crossentropy_parameter_search routine and get dictionary 
    # for each model architecture/configuration.
    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)

    # 2. Plot Train and Validation losses for each 
    # model all on single plot (it should be 10 lines total).
    #         x-axis should be epochs, y-axis should me 
    #         Crossentropy loss, REMEMBER to add legend
    plt.figure(1)
    best_model = None # used for (3)
    best_model_name = None
    best_loss = np.inf

    # note: ce_configs is a dictionary, 
    # which is the KEY - VALUE pairs (a Map)
    for model_name, model_params in ce_configs.items():
        validation_loss = model_params["val"]
        if min(validation_loss) < best_loss: # choose model w/ lowest validation loss for (3)
            best_loss = min(validation_loss)
            best_model_name = model_name
            best_model = model_params.get("model")
        plt.plot(model_params["train"], label=f"{model_name}" + " train")
        plt.plot(model_params["val"], label= f"{model_name}" + " val")
    plt.title("X-entropy loss over NN for train and validation")
    plt.xlabel("Epochs")
    plt.ylabel("X-entropy loss")
    plt.legend()

    # 3. Choose and report the best model configuration 
    # based on validation losses. In particular you should 
    # choose a model that achieved the lowest validation 
    # loss at ANY point during the training.
    print("Best model: ", best_model_name)
    print("Validation loss: ", best_loss)

    # 4. Plot best model guesses on test set (using plot_model_guesses 
    #                                      function from train file)
    plt.figure(2)
    plot_model_guesses(DataLoader(dataset_test), best_model, "best model NN Predictions in X-entropy")
    plt.show()

    # accuracy on test
    print("Accuracy - ", accuracy_score(best_model, DataLoader(dataset_test, batch_size=32, shuffle=False)))

if __name__ == "__main__":
    main()
