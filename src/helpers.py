import math
import torch
import numpy as np
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import os
import multiprocessing
import matplotlib.pyplot as plt

def denormalize(images):
    # invTrans = transforms.Compose(
    #     [
    #         transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
    #         transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
    #     ]
    # )
    """Transform images from [-1.0, 1.0] to [0, 255] and cast them to uint8."""
    return ((images + 1.) / 2. * 255).astype(np.uint8)

def visualize_one_batch(dataLoader, max_n: int = 5):
    """
    Visualize one batch of data.

    :param dataLoaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    # obtain one batch of training images
    # First obtain an iterator from the train dataloader
    #dataiter  = iter(dataLoaders["train"])
    dataiter = iter(dataLoader)

    rimages, labels = dataiter.next()

    # Get class names from the train data loader
    traindata = dataLoader.dataset
    # Get class name from the folder names (which are stored in traindata.classes)
    #class_names  = [name.split('.')[1] for name in traindata.classes]

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(rimages, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        # convert from torch to numpy object, then denormalize
        img = denormalize(images[idx].numpy())
        ax.imshow(img)
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        #ax.set_title(class_names[labels[idx].item()])

    return rimages
