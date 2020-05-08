import torch
import matplotlib.pyplot as plt
import numpy as np


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
        print('GPU is not available in your computer. CPU will be used instead')
    return device


def imshow_tensor(image, ax=None, mean=0, std=1, title=""):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')
    plt.title(title)

    return ax, image
