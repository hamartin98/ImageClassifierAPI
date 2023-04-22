import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms

'''Collection of image transformation related utility functions'''


def imshow(image):
    '''Show image'''

    image = image / 2 + 0.5     # unnormalize
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def splitImage(image, rows: int, cols: int) -> None:
    '''Split the image to specified rows and columns'''

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


def splitImageToTensors(image, rows: int, cols: int) -> None:
    '''Split the image and covert to tensors'''

    result = []

    parts = splitImage(image, rows, cols)
    for row in range(0, rows):
        resultRow = []
        for col in range(0, cols):
            imageTensor = imageToTensor(parts[row][col])
            resultRow.append(imageTensor)
        result.append(resultRow)

    return result


def imageToTensor(image) -> None:
    '''Convert the given image to tensor'''

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
