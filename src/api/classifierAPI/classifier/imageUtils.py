import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


def imshow(image):
    image = image / 2 + 0.5     # unnormalize
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def splitImage(image, rows, cols) -> None:
    result = []

    height, width, channels = image.shape
    verticalSize = height // rows
    horizontalSize = width // cols

    for row in range(0, rows):
        resultRow = []

        for col in range(0, cols):
            vertical = row * verticalSize
            horizontal = col * horizontalSize

            part = image[vertical: vertical + verticalSize,
                         horizontal: horizontal + horizontalSize]
            resultRow.append(part)
        result.append(resultRow)

    return result


def imageToTensor(image) -> None:
    transformedImage = image

    imageTensor = torch.from_numpy(transformedImage)
    imageTensor = imageTensor.permute(2, 0, 1)
    imageTensor = imageTensor.numpy()
    imageTensor = np.transpose(imageTensor, (1, 2, 0))

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    imageTensor = transform(imageTensor)
    imageTensor = torch.from_numpy(np.expand_dims(imageTensor, axis=0))

    return imageTensor
