# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = torch.sqrt(torch.tensor(1.0 / d))

        self.W0 = Parameter(torch.empty(d, h))
        self.b0 = Parameter(torch.empty(h))
        self.W1 = Parameter(torch.empty(h, k))
        self.b1 = Parameter(torch.empty(k))
       

        # fill in parameters with randomiized values based on UNIF(-1/sqrt(d), 1/sqrt(d))
        distribution = Uniform(-alpha, alpha)
        self.W0.data = distribution.sample(self.W0.shape)
        self.b0.data = distribution.sample(self.b0.shape)
        self.W1.data = distribution.sample(self.W1.shape)
        self.b1.data = distribution.sample(self.b1.shape)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        
        return (relu((x @ self.W0) + self.b0) @ self.W1) + self.b1
        


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = torch.sqrt(torch.tensor(1.0 / d))
        self.W0 = Parameter(torch.empty(d, h0))
        self.b0 = Parameter(torch.empty(h0))
        self.W1 = Parameter(torch.empty(h0, h1))
        self.b1 = Parameter(torch.empty(h1))
        self.W2 = Parameter(torch.empty(h1, k))
        self.b2 = Parameter(torch.empty(k))

        distribution = Uniform(-alpha, alpha)
        self.W0.data = distribution.sample(self.W0.shape)
        self.b0.data = distribution.sample(self.b0.shape)
        self.W1.data = distribution.sample(self.W1.shape)
        self.b1.data = distribution.sample(self.b1.shape)
        self.W2.data = distribution.sample(self.W2.shape)
        self.b2.data = distribution.sample(self.b2.shape)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(relu(x @ self.W0 + self.b0) @ self.W1 + self.b1) @ self.W2  + self.b2


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    loss_list = []
    accuracy = 0
    model.train()
    while accuracy < 0.99:

        epoch_loss = 0
        for samples, expected in train_loader:
            optimizer.zero_grad()
            prediction = model(samples)
            loss = cross_entropy(prediction, expected)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_ratio = epoch_loss / len(train_loader)
        loss_list.append(loss_ratio)
        accuracy = accuracy_score(model, train_loader) 
    return loss_list



def accuracy_score(model, train_loader):
    total_correct = 0
    total_samples = 0
    for sample, expected in train_loader:
        prediction = model(sample)
        data = prediction.data
        _, predicted = torch.max(data, dim=1)
        total_samples += expected.shape[0]
        for pred, exp in zip(predicted, expected):
            total_correct += (pred == exp).sum().item()  
    return total_correct / total_samples

@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    

    # load train and test sets
    dataset_train = TensorDataset(x, y)
    dataset_test = TensorDataset(x_test, y_test)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    # F1 model
    f1 = F1(64, 784, 10)
    optimizer = Adam(f1.parameters(), lr=1e-3)
    # 1. Train a model
    F1_loss = train(f1, optimizer, train_loader)
    # 2. Plot per epoch losses
    plt.plot(F1_loss)
    plt.title("F1 Loss")
    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropy Loss")
    plt.show()
    
    test_loss = 0
    for sample, expected in test_loader:
        prediction = f1(sample)
        loss = cross_entropy(prediction, expected)
        test_loss += loss.item()
    # 4. Report total number of parameters for each network
    loss_ratio = test_loss / len(test_loader)
    print("F1 Test Loss - ", loss_ratio)
    total_param = 0
    for param in f1.parameters():
        total_param += param.numel()
    print("F1 Parameters - ", total_param)

    #  3. Report accuracy and loss on test set
    print("Accuracy (F1) - ", accuracy_score(f1, test_loader))



    # F2 Model
    f2 = F2(32, 32, 784, 10)
    optimizer = Adam(f2.parameters(), lr=1e-3)
    # 1. Train a model
    F2_loss = train(f2, optimizer, train_loader)
    # 2. Plot per epoch losses
    plt.plot(F2_loss)
    plt.title("F2 Loss")
    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropy Loss")
    plt.show()
    
    # 4. Report total number of parameters for each network
    test_loss = 0
    for sample, expected in test_loader:
        prediction = f2(sample)
        loss = cross_entropy(prediction, expected)
        test_loss += loss.item()
    
    ratio_F2 = test_loss / len(test_loader)
    print("F2 Test Loss Ratio - ", ratio_F2)
    total_param_F2 = 0
    for param in f2.parameters():
        total_param_F2 += param.numel()
    print("F2 Parameters - ", total_param_F2)

    #  3. Report accuracy and loss on test set
    print("Accuracy (F2) - ", accuracy_score(f2, test_loader))


if __name__ == "__main__":
    main()
