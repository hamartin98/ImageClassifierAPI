import cv2
import numpy as np
from typing import Dict

import torch

from .classificationMap import ClassificationMap, BaseClassification
from .config.config import Config
from .config.classifierConfig import ClassifierConfig
from .models.baseNetwork import BaseNetwork
from .utils.imageUtils import splitImageToTensors


class MultiModelClassifier:
    '''Classify images with multiple models'''

    def __init__(self) -> None:
        '''Basic initialization'''

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
        '''Setup each classifier'''

        classifications: Dict[str,
                              BaseClassification] = self.classifications.getClassifications()

        for key, classification in classifications.items():
            classification.configureAndSetupNetwork(self.baseConfig)

    def dataSetup(self, image, rows: int, cols: int) -> None:
        '''Prepare data'''

        self.rows = rows
        self.cols = cols
        self.originalData = image

        self.prepareImage()

    def prepareImage(self) -> None:
        '''Prepare the given image to classification'''

        transformedImage = np.asarray(
            bytearray(self.originalData.read()), dtype="uint8")
        transformedImage = cv2.imdecode(transformedImage, cv2.IMREAD_COLOR)

        mean = tuple(self.baseConfig.getMean())
        std = tuple(self.baseConfig.getStd())
        
        self.preparedData = splitImageToTensors(
            transformedImage, self.rows, self.cols, mean, std)

    def classifyWithMultiModels(self, image, rows: int, cols: int) -> None:
        '''Classify images with multiple models'''

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

                        result[row][col][key] = int(res)

        return result

    def createResponseSkeleton(self, rows: int, cols: int) -> None:
        '''Create response dictionary with the given rows and columns'''

        result = [[dict() for row in range(rows)] for col in range(cols)]
        return result
