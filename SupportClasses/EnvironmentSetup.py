import torch
import torchvision
import random
from itertools import permutations
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

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
    # Convert this to a set to give only the unique values. Then convert back to a list so we can slice it.
    return list(set(list(permutations(DATA_DISTRIBUTION[number_of_nodes], number_of_nodes))))[:number_of_nodes]


def filter_data_by_classes(num_of_classes, trainset, testset):
    """
    Filters a dataset and testset by the classes we wish to use for classification.
    :param num_of_classes: The number of classes to select.
    :param trainset: The training set.
    :param testset: The testing set.
    :return: The training, validation and testing set.
    """
    # Select unique num_of_classes at random from the set of all classes.
    # classes = random.sample(list(trainset.class_to_idx.values()), k=num_of_classes)
    # Get the first four classes
    classes = list(trainset.class_to_idx.values())[:num_of_classes]
    print(f"The classes in the training and testing set are {trainset.classes[:num_of_classes]}")

    # Reduce the dataset to only contain the classes we want.
    train_idx = [target in classes for target in trainset.targets]
    test_idx = [target in classes for target in testset.targets]

    # CIFAR10 and MNIST datasets differ here
    if not isinstance(trainset.targets, list):
        trainset.targets, testset.targets = trainset.targets[train_idx], testset.targets[test_idx]
        trainset.data, testset.data = trainset.data[train_idx], testset.data[test_idx]
    else:
        trainset.targets, testset.targets = ([target for target, val in zip(trainset.targets, train_idx) if val == True],
                                             [target for target, val in zip(testset.targets, test_idx) if val == True])
        trainset.data, testset.data = ([data for data, val in zip(trainset.data, train_idx) if val == True],
                                       [data for data, val in zip(testset.data, test_idx) if val == True])

    # Split into testing and validation set
    trainset, valset = torch.utils.data.random_split(trainset,
                                                     [round(len(trainset) * 0.85), round(len(trainset) * 0.15)])

    return trainset, valset, testset, classes


def download_mnist(num_of_classes, train_batch_size=25, test_batch_size=100, num_workers=2):
    """
    Downloads the MNIST dataset.
    :param num_of_classes: The number of classes to select from the dataset.
    :param train_batch_size: Specifies the training batch size.
    :param test_batch_size: Specifies the test batch size.
    :param num_workers: Specifies the number of workers to load data. Modify this value based on your system resources.
    :return: Returns the value of filter_data_by_classes.
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    return filter_data_by_classes(num_of_classes, trainset, testset)


def download_fashion_mnist(num_of_classes, train_batch_size=25, test_batch_size=100, num_workers=2):
    """
    Downloads the MNIST dataset.
    :param num_of_classes: The number of classes to select from the dataset.
    :param train_batch_size: Specifies the training batch size.
    :param test_batch_size: Specifies the test batch size.
    :param num_workers: Specifies the number of workers to load data. Modify this value based on your system resources.
    :return: Returns the value of filter_data_by_classes.
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                 download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)

    return filter_data_by_classes(num_of_classes, trainset, testset)


def download_cifar10(num_of_classes, train_batch_size=25, test_batch_size=100, num_workers=2):
    """
    Downloads the ImageNet dataset.
    :param num_of_classes: The number of classes to select from the dataset.
    :param train_batch_size: Specifies the training batch size.
    :param test_batch_size: Specifies the test batch size.
    :param num_workers: Specifies the number of workers to load data. Modify this value based on your system resources.
    :return: Returns the value of filter_data_by_classes.
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    return filter_data_by_classes(num_of_classes, trainset, testset)


def download_dataset(num_of_classes, dataset):
    if dataset == 'cifar10':
        return download_cifar10(num_of_classes)
    elif dataset == 'MNIST':
        return download_mnist(num_of_classes)
    return download_fashion_mnist(num_of_classes)


def create_single_loader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True, num_workers=2)


def create_data_loaders(training_sets, validation_set, test_set):
    """
    Returns data loaders based on the training, validation and testing set passed in.
    :param train_set: The training data.
    :param validation_set: The validation data.
    :param test_set: The testing data.
    :return:
    """
    training_loaders = [torch.utils.data.DataLoader(train_set, batch_size=25,
                                               shuffle=True, num_workers=2)
                        for train_set in training_sets]
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=100,
                                                    shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                              shuffle=True, num_workers=2)
    return training_loaders, validation_loader, test_loader


def unbalance_training_set(train_set, classes, data_distribution, subset_max_size):
    """
    This method will be used to unbalance a dataset.
    :return:
    """
    unbalanced_dataset = []
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True, num_workers=2)

    # Let's separate the training set based on class
    for dist, target in zip(data_distribution, classes):

        for inputs, labels in data_loader:
            # Get only the data that belongs to the class we are looking at.
            idx = (labels == target)
            data, labels = inputs[idx], labels[idx]

            # Reduce the size of the training set to 100 images per class
            temp = TensorDataset(data, labels)
            temp, _ = torch.utils.data.random_split(temp, [subset_max_size, len(temp) - subset_max_size])

            # Now take a subset of this set
            first, second = round(len(temp) * dist), round(len(temp) * (1 - dist))
            rounding_error = (first+second)-len(temp)
            class_subset, _ = torch.utils.data.random_split(temp,
                                          [first, second-rounding_error])

            unbalanced_dataset.append(class_subset)
            break

    return ConcatDataset(unbalanced_dataset)
