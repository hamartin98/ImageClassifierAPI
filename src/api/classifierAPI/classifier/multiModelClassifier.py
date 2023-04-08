from typing import Dict

import torch
import cv2
import numpy as np

from config import Config
from classifierConfig import ClassifierConfig
from models.baseNetwork import BaseNetwork
from classificationMap import ClassificationMap, BaseClassification

from imageUtils import splitImageToTensors


class MultiModelClassifier:
    def __init__(self) -> None:
        self.baseConfig = ClassifierConfig(Config.getPath())

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')

        self.rows = 1
        self.cols = 1
        self.originalData = None
        self.preparedData = None
        self.classifications = ClassificationMap()

        self.setupClassifiers()

    def setupClassifiers(self) -> None:
        classifications: Dict[str,
                              BaseClassification] = self.classifications.getClassifications()

        for key, classification in classifications.items():
            classification.configureAndSetupNetwork(
                self.baseConfig, self.device)

    def dataSetup(self, image, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.originalData = image

        self.prepareImage(image, rows, cols)

        self.setupClassifiers()

    def prepareImage(self) -> None:

        transformedImage = np.asarray(
            bytearray(self.originalData.read()), dtype="uint8")
        transformedImage = cv2.imdecode(transformedImage, cv2.IMREAD_COLOR)

        self.preparedData = splitImageToTensors()

    def classifyWithMultiModels(self, image, rows: int, cols: int) -> None:
        result = self.createResponseSkeleton(rows, cols)

        self.dataSetup(image, rows, cols)

        classifications: Dict[str,
                              BaseClassification] = self.classifications.getClassifications()

        for key, classification in classifications.items():
            network: BaseNetwork = classification.getNetwork()
            network.eval()

            for row in range(0, self.rows):
                for col in range(0, self.cols):
                    res = None

                    with torch.no_grad():
                        imageTensor = self.preparedData[row][col]
                        output = network(imageTensor)
                        _, predictions = torch.max(output, 1)
                        res = predictions[0]
                        print(f'Class: {res}')

                        result[row][col][key] = int(res)

        return result

    def createResponseSkeleton(self, rows: int, cols: int) -> None:
        rowItem = {'building': 0, 'vegetation': 0, 'road': 0}
        result = [[rowItem for row in range(rows)] for col in range(cols)]
        return result
