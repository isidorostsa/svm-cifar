import time

class Timer:
    def __enter__(self, verbose=True):
        self.start = time.time()
        self.verbose = verbose
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.verbose:
            print(f"Elapsed time: {self.interval:.3f} seconds")

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def fetch_datasets(transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(), # this converts from (H, W, C) to (C, H, W) and scales to [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # this normalizes the mean and std to be 0.5
            transforms.RandomHorizontalFlip(p=0.5), # the following are data augmentation 
            transforms.RandomRotation(15) # this helps generalization
        ])
    
    # load the dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    return trainset, testset

def get_loaders(batch_size=128, batch_size_test=128, transform=None):
    trainset, testset = fetch_datasets(transform)

    # create a dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=4)

    return trainloader, testloader

def show_reconstruction(pca, x_test, y_test, classes, n=2, index=0):
    n = min(n, len(x_test))

    x_test_pca = pca.transform(x_test[index:index+n])
    y_test = y_test[index:index+n]
    x_test_reconstructed = pca.inverse_transform(x_test_pca)

    # normalize the reconstructed data
    x_max = x_test_reconstructed.max()
    x_min = x_test_reconstructed.min()
    x_test_reconstructed = (x_test_reconstructed - x_min) / (x_max - x_min)

    show_images(x_test_reconstructed.reshape(-1, 32, 32, 3), y_test, classes, n)



def show_images(images, labels, classes, n_images=6, grayscale=False):
    """
    Shows n_images images from the dataset
    """
    fig, axes = plt.subplots(nrows=1, ncols=n_images, figsize=(n_images*2, 4))
    
    for i in range(n_images):
        if grayscale:
            axes[i].imshow(images[i], cmap='gray')
        else:
            axes[i].imshow(images[i])
        axes[i].set_title(classes[labels[i]])
        axes[i].axis('off')
        
    plt.show()

def plot_accuracy_loss(history):
    """
    Plots the training and test accuracy
    """
    train_acc = history['train_accuracy']
    test_acc = history['test_accuracy']
    train_loss = history['train_loss']
    test_loss = history['test_loss']

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.grid(visible=True)
    plt.xticks(np.arange(0, len(train_acc)+1, 5))
    plt.ylim([0, 1])
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.grid(visible=True)
    plt.xticks(np.arange(0, len(train_loss)+1, 5))
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()


def compare_accuracy_and_overfitting(histories):
    """
    input: test_accs: dictionary of test accuracies in 
    {"model_name": [test_acc...] per epoch, ...}
    """

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.grid(visible=True)

    lens = [len(history['test_accuracy']) for history in histories.values()]
    for label, history in histories.items():
        plt.plot(history['test_accuracy'], label=label)


    plt.xticks(np.arange(0, max(lens)+1, 5))    
    plt.title('Test Accuracy')
    plt.legend()
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.05))

    plt.subplot(1, 2, 2)
    plt.title('Overfitting Ratio (Test Loss / Train Loss)')

    for history in histories.values():
        overfitting_ratio = []
        for (test_loss, train_loss) in zip(history['test_loss'], history['train_loss']):
            overfitting_ratio.append(test_loss/train_loss)

        history['overfitting_ratio'] = overfitting_ratio

    for label, history in histories.items():
        plt.plot(history['overfitting_ratio'], label=label)
    plt.grid(visible=True)
    plt.xticks(np.arange(0, max(lens)+1, 5))

    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Overfitting Ratio')


def compare_accuracy_and_overfitting_and_time(histories):
    """
    input: test_accs: dictionary of test accuracies in 
    {"model_name": [test_acc...] per epoch, ...}
    """

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.grid(visible=True)

    lens = [len(history['test_accuracy']) for history in histories.values()]
    for label, history in histories.items():
        plt.plot(history['test_accuracy'], label=label)


    plt.xticks(np.arange(0, max(lens)+1, 5))    
    plt.title('Test Accuracy')
    # legend at top split in 2 columns
    plt.legend(ncol=2, loc='upper center')
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.05))

    plt.subplot(2, 2, 2)
    plt.title('Overfitting Ratio (Test Loss / Train Loss)')

    for history in histories.values():
        overfitting_ratio = []
        for (test_loss, train_loss) in zip(history['test_loss'], history['train_loss']):
            overfitting_ratio.append(test_loss/train_loss)

        history['overfitting_ratio'] = overfitting_ratio

    for label, history in histories.items():
        plt.plot(history['overfitting_ratio'], label=label)
    plt.grid(visible=True)
    plt.xticks(np.arange(0, max(lens)+1, 5))

    # legend at top left
    plt.legend(loc='upper left')

    # plot the loss
    plt.subplot(2, 2, 3)
    plt.title('Loss')

    for label, history in histories.items():
        plt.plot(history['test_loss'], label=label)
    plt.grid(visible=True)
    plt.xticks(np.arange(0, max(lens)+1, 5))

    # legend at top right split in 2 columns
    plt.legend(ncol=2, loc='upper right')

    plt.subplot(2, 2, 4)
    plt.title('Time per Epoch') # bar chart
    # the labels are too long, so we need to rotate them

    for label, history in histories.items():
        plt.bar(label, history['time_per_epoch'])

    # rotate the labels to be readable
    plt.xticks(rotation=90)

    plt.show()

def compare_accuracy_and_overfitting_and_time_and_parameters(histories, models):
    """
    input: test_accs: dictionary of test accuracies in 
    {"model_name": [test_acc...] per epoch, ...}
    """

    plt.figure(figsize=(20, 20))
    plt.subplot(3, 2, 1)
    plt.grid(visible=True)

    lens = [len(history['test_accuracy']) for history in histories.values()]
    for label, history in histories.items():
        plt.plot(history['test_accuracy'], label=label)


    plt.xticks(np.arange(0, max(lens)+1, 5))    
    plt.title('Test Accuracy')
    # legend at top split in 2 columns
    plt.legend(ncol=2, loc='upper center')
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1.1, 0.05))

    plt.subplot(3, 2, 2)
    plt.title('Overfitting Ratio (Test Loss / Train Loss)')

    for history in histories.values():
        overfitting_ratio = []
        for (test_loss, train_loss) in zip(history['test_loss'], history['train_loss']):
            overfitting_ratio.append(test_loss/train_loss)

        history['overfitting_ratio'] = overfitting_ratio

    for label, history in histories.items():
        plt.plot(history['overfitting_ratio'], label=label)
    plt.grid(visible=True)
    plt.xticks(np.arange(0, max(lens)+1, 5))

    # legend at top left
    plt.legend(loc='upper left')

    # plot the loss
    plt.subplot(3, 2, 3)
    plt.title('Loss')

    for label, history in histories.items():
        plt.plot(history['test_loss'], label=label)
    plt.grid(visible=True)
    plt.xticks(np.arange(0, max(lens)+1, 5))

    # legend at top right split in 2 columns
    plt.legend(ncol=2, loc='upper right')

    plt.subplot(3, 2, 4)
    plt.title('Time per Epoch') # bar chart
    # the labels are too long, so we need to rotate them

    for label, history in histories.items():
        plt.bar(label, history['time_per_epoch'])

    # rotate the labels to be readable
    plt.xticks(rotation=90)

    for (label, model) in models.items():
        params = get_parameter_count(model)

        if "pixel" in label:
            params = params - DIM*DIM*CHANNELS*DIM*DIM + DIM*DIM*CHANNELS
        histories[label]['params'] = params

    plt.subplot(3, 2, 5)
    plt.title('Number of Parameters') # bar chart
    # this plot is a bar chart, sideways
    # the labels are on the y axis in the left side
    # the values are on the x axis in the bottom

    for label, history in histories.items():
        plt.barh(label, history['params'])

    plt.show()

def get_parameter_count(model):
    """
    Returns the number of parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
