import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import os

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


def splitImageToTensors(image, rows: int, cols: int, transformations) -> None:
    '''Split the image and covert to tensors'''

    result = []

    parts = splitImage(image, rows, cols)
    for row in range(0, rows):
        resultRow = []
        for col in range(0, cols):
            imageTensor = imageToTensor(parts[row][col])
            imageTensor = transformations(imageTensor)

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

    transform = transforms.Compose([transforms.ToTensor()])
    imageTensor = transform(imageTensor)
    imageTensor = torch.from_numpy(np.expand_dims(imageTensor, axis=0))

    return imageTensor


def calculateMeanAndStdForImages(path: str):
    '''Calculate mean and standard deviation values from the given images'''

    imageFilesDir = Path(path)
    files = list(imageFilesDir.rglob('*.jpg'))

    mean = np.array([0.0, 0.0, 0.0])
    stdTemp = np.array([0.0, 0.0, 0.0])
    std = np.array([0.0, 0.0, 0.0])

    fileNum = len(files)

    if fileNum == 0:
        raise Exception('No files found')

    for idx in range(fileNum):
        path = os.path.relpath(files[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(float) / 255.0

        for channel in range(3):
            mean[channel] += np.mean(image[:, :, channel])

    mean = (mean / fileNum)

    for idx in range(fileNum):
        path = os.path.relpath(files[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(float) / 255.0

        divider = (image.shape[0] * image.shape[1])
        for channel in range(3):
            data = image[:, :, channel]
            stdTemp[channel] += ((data - mean[channel]) ** 2).sum() / divider

    std = np.sqrt(stdTemp / fileNum)

    print(f'Mean: {mean}\n Std: {std}')

    return {
        'mean': mean,
        'std': std
    }
