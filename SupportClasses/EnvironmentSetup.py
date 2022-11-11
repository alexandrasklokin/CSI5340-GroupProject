import torch
import torchvision
from itertools import permutations

"""
The following methods are to be used to download and distribute the dataset among the various nodes.
"""

DATA_DISTRIBUTION = [None, None, (0.25, 0.75), (0.25, 0.25, 0.50), (0.15, 0.15, 0.15, 0.55)]


def data_distribution(number_of_nodes):
    """
    This method returns a list of tuples of length number_of_nodes containing the data distribution per-client.
    :param number_of_nodes: The number of devices in the federated model.
    :return: A list of tuples containing all unique permutations of the size number_of_nodes.
    """
    return_list = []
    for elem in permutations(DATA_DISTRIBUTION[number_of_nodes], number_of_nodes):
        if elem not in return_list:
            return_list.append(elem)
    return return_list[:number_of_nodes]


def download_mnist(train_batch_size=25, test_batch_size=100, num_workers=2):
    """
    Downloads the MNIST dataset.
    :param train_batch_size: Specifies the training batch size.
    :param test_batch_size: Specifies the test batch size.
    :param num_workers: Specifies the number of workers to load data. Modify this value based on your system resources.
    :return: Returns a tuple containing (train_loader, test_loader).
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset,
                                                     [round(len(dataset)*0.85), round(len(dataset)*0.15)])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=25,
                                               shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(valset, batch_size=100,
                                                    shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                              shuffle=True, num_workers=2)

    return train_loader, validation_loader, test_loader


def download_imagenet(train_batch_size=25, test_batch_size=100, num_workers=2):
    """
    Downloads the MNIST dataset.
    :param train_batch_size: Specifies the training batch size.
    :param test_batch_size: Specifies the test batch size.
    :param num_workers: Specifies the number of workers to load data. Modify this value based on your system resources.
    :return: Returns a tuple containing (train_loader, test_loader).
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.ImageNet(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=25,
                                               shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageNet(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                              shuffle=True, num_workers=2)

    return train_loader, test_loader


# TODO: Implement this function
def unbalance_datasets():
    """
    This method will be used to unbalance a dataset.
    :return:
    """
    return None
