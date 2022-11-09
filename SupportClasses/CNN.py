import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """
    Here we create the CNN model. It will consist of two convolutional layers, with max pooling in between,
    and one fully connected linear layer.
    """
    def __init__(self, numberOfClasses):
        """
        General class for creating a CNN. Includes the numberOfClasses parameter.
        :param numberOfClasses: Indicates the number of classes in the classification task.
        """
        super(ConvNet, self).__init__()
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

    for inputs, labels in data_loader:
        # send the inputs and labels to the same device
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accuracy on the training set
        _, predictions = torch.max(outputs.data, 1)
        acc = (predictions==labels).sum().item()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


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
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)

            # Get the outputs and the loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accuracy on the testing set
            _, predictions = torch.max(outputs.data, 1)
            acc = (predictions == labels).sum().item()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)