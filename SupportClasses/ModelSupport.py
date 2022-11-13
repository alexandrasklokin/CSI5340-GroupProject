import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from collections import OrderedDict


class FederatedModel:
    """
    This class contains all information relevant to a Federated Model in one place.
    """
    def __init__(self, train_loader, validation_loader, model):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.model = model

    def get_len_train_data(self):
        return len(self.train_loader.dataset)


class ConvNetMnist(nn.Module):
    """
    Here we create the ms model. It will consist of two convolutional layers, with max pooling in between,
    and one fully connected linear layer.
    """
    def __init__(self, numberOfClasses):
        """
        General class for creating a ms. Includes the numberOfClasses parameter.
        :param numberOfClasses: Indicates the number of classes in the classification task.
        """
        super(ConvNetMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, numberOfClasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNetCifar(nn.Module):
    def __init__(self, numberOfClasses):
        super(ConvNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, numberOfClasses)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, data_loader, optimizer, criterion, device):
    """
    The following method is used to train a model on a dataset.
    :param model: The model to be used.
    :param data_loader: The data_loader.
    :param optimizer: The optimizer, used to optimize the model.
    :param criterion: The loss criterion.
    :param device: The device we wish to perform the computations (CPU, GPU)
    :return: Returns a tuple containing (training_loss, training_accuracy) for the current epoch.
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    # These are for debugging
    # count = 0
    # acc_1 = 0
    # acc_2 = 0
    # acc_3 = 0

    for inputs, labels in data_loader:
        # Used for debugging
        # count += 25
        # acc_1 += list(labels).count(0)
        # acc_2 += list(labels).count(1)
        # acc_3 += list(labels).count(2)
        # send the inputs and labels to the same device
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accuracy on the training set
        _, predictions = torch.max(outputs.data, 1)
        epoch_loss += loss.item()
        epoch_acc += (predictions==labels).sum().item()

    # Used for debugging
    # print(count, acc_1, acc_2, acc_3)
    return epoch_loss / len(data_loader.dataset), epoch_acc / len(data_loader.dataset)


def test(model, data_loader, criterion, device):
    """
    The following method is used to test a model on a dataset.
    :param model: The model to be used.
    :param data_loader: The data_loader.
    :param criterion: The loss criterion.
    :param device: The device we wish to perform the computations (CPU, GPU)
    :return: Returns a tuple containing (testing_loss, testing_accuracy) for the current epoch.
    """
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            # send the inputs and labels to the same device
            inputs, labels = inputs.to(device), labels.to(device)

            # Get the outputs and the loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accuracy on the testing set
            _, predictions = torch.max(outputs.data, 1)
            epoch_loss += loss.item()
            epoch_acc += (predictions == labels).sum().item()
    return epoch_loss / len(data_loader.dataset), epoch_acc / len(data_loader.dataset)


def federated_averaging(fed_models: Dict[str, FederatedModel]):
    """
    This method takes in a dictionary of federated models and averages their weights to perform FED_AVG.
    :param fed_models: This a federated model containing the data and model.
    :param total_data_size: This is the total amount of data over all systems.
    :return: The averaged weights across all models.
    """
    average_weights = OrderedDict()
    total_data_size = sum([fed_model.get_len_train_data() for fed_model in fed_models.values()])

    for i, fed_model in enumerate(fed_models.values()):
        weight_coefficient = round(fed_model.get_len_train_data() / total_data_size, 3)
        local_weights = fed_model.model.state_dict()

        # Iterate over the weights in the local networks and add them to a new ordered dictionary
        for key in local_weights.keys():
            if i == 0:
                average_weights[key] = weight_coefficient * local_weights[key]
            else:
                average_weights[key] += weight_coefficient * local_weights[key]
    return average_weights
