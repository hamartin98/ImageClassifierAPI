import torch
import torchvision.transforms as transforms

import cv2
import numpy as np
import os

from .config import Config
from .classifierConfig import ClassifierConfig
from .models.FirstNetwork import FirstNetwork

from .imageUtils import splitImage, imageToTensor


class ImageClassifier():
    def __init__(self) -> None:
        #MODEL_PATH = os.path.abspath('/data/models/model_latest.pth') # For docker
        #MODEL_PATH = os.path.relpath('data/models/model_latest.pth')

        config = ClassifierConfig(Config.getPath())
        if not os.path.exists(config.getModelPath()):
            print('Path not found')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')

        self.network = FirstNetwork()
        self.network.load_state_dict(torch.load(config.getModelPath(), map_location = device))

    def classifyImage(self, image) -> None:
        print(image)
        transformedImage = image

        transformedImage = np.asarray(bytearray(image.read()), dtype="uint8")
        transformedImage = cv2.imdecode(transformedImage, cv2.IMREAD_COLOR)

        imgTensor = torch.from_numpy(transformedImage)
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = imgTensor.numpy()
        imgTensor = np.transpose(imgTensor, (1, 2, 0))

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        imgTensor = transform(imgTensor)
        imgTensor = torch.from_numpy(np.expand_dims(imgTensor, axis=0))

        output = None
        self.network.eval()
        with torch.no_grad():
            output = self.network(imgTensor)
            _, predictions = torch.max(output, 1)
            print(f'Class: {predictions[0]}')

        return predictions[0]

    def classifyImageParts(self, image) -> None:
        print(image)
        transformedImage = image

        transformedImage = np.asarray(bytearray(image.read()), dtype="uint8")
        transformedImage = cv2.imdecode(transformedImage, cv2.IMREAD_COLOR)
        parts = splitImage(transformedImage, 10, 10)

        transformedImage = parts[0][0]

        imgTensor = torch.from_numpy(transformedImage)
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = imgTensor.numpy()
        imgTensor = np.transpose(imgTensor, (1, 2, 0))

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        imgTensor = transform(imgTensor)
        imgTensor = torch.from_numpy(np.expand_dims(imgTensor, axis=0))

        output = None
        self.network.eval()
        with torch.no_grad():
            output = self.network(imgTensor)
            _, predictions = torch.max(output, 1)
            print(f'Class: {predictions[0]}')

        return predictions[0]

    def splitAndClassify(self, image, rows, cols) -> None:
        result = []
        transformedImage = np.asarray(bytearray(image.read()), dtype="uint8")
        transformedImage = cv2.imdecode(transformedImage, cv2.IMREAD_COLOR)

        parts = splitImage(transformedImage, rows, cols)

        self.network.eval()

        for row in range(0, len(parts)):
            resultRow = []
            for col in range(0, len(parts[row])):
                res = None
                with torch.no_grad():
                    imageTensor = imageToTensor(parts[row][col])
                    output = self.network(imageTensor)
                    _, predictions = torch.max(output, 1)
                    res = predictions[0]
                    print(f'Class: {res}')

                    resultRow.append(
                        {'building': int(res), 'vegetation': 0, 'road': 0})
                    # resultRow.append(int(res))

            result.append(resultRow)

        return result
