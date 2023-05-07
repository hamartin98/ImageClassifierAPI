import cv2
import glob
import numpy as np
import os
from typing import Any, List

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CustomDataset(Dataset):
    '''Custom dataset to handle custom image data'''

    def __init__(self, path: str, classes: tuple, imgDim=(62, 62), transformList=[], normalizerTransform=None, balanceClasses=False):
        '''Default initalization'''

        self.normalizerTransform = transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if normalizerTransform:
            self.normalizerTransform = normalizerTransform

        self.balanceClasses = balanceClasses
        self.transformList: List[Any] = transformList
        self.transformList.append(self.normalizerTransform)
        self.composedTransforms = transforms.Compose(transformList)
        self.imagesPath = path
        targetDataSetSize = 0

        fileList = glob.glob(os.path.join(self.imagesPath, "*"))

        self.data = []
        perClassData = []

        for classPath in fileList:
            classPath = os.path.relpath(classPath)
            className = os.path.normpath(classPath).split(os.path.sep)[-1]
            paths = glob.glob(classPath + "/*.jpg")
            currentData = []
            for imagePath in paths:
                currentData.append([os.path.join(imagePath), className])
            perClassData.append(currentData)

            currentClassSize = len(paths)
            if currentClassSize > targetDataSetSize:
                targetDataSetSize = currentClassSize

        for currentData in perClassData:
            multiplier = 1

            if self.balanceClasses:
                multiplier = round(targetDataSetSize / len(currentData))

            self.data.extend(currentData * multiplier)

        self.createClassMap(classes)
        self.imgDim = imgDim

    def __len__(self):
        '''Return dataset size'''

        return len(self.data)

    def __getitem__(self, idx: int):
        '''Return data from the given index'''

        img_path, className = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.imgDim)

        imgTensor = torch.from_numpy(img)
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = imgTensor.numpy()
        imgTensor = np.transpose(imgTensor, (1, 2, 0))

        classId = self.classMap[className]
        classId = torch.tensor(classId)

        if self.composedTransforms:
            imgTensor = self.composedTransforms(imgTensor)

        return imgTensor, classId

    def createClassMap(self, classes: tuple):
        '''Create class map'''

        classMap = {}
        for idx, cls in enumerate(classes):
            classMap[cls] = idx

        self.classMap = classMap

    def addTransforms(self, transformations: List[Any]) -> None:
        '''Extend list of transformations of the dataset'''

        if self.transformList is None:
            self.transformList = []

        self.transformList.extend(transformations)
        self.composedTransforms = transforms.Compose(self.transformList)

    def overrideTransforms(self, transformations: List[Any]) -> None:
        '''Override the list of transformations of the dataset'''

        self.transformList = transformations
        transformations.append(self.normalizerTransform)
        self.composedTransforms = transforms.Compose(self.transformList)

    def setNormalizerTransform(self, normalizerTransform) -> None:
        self.normalizerTransform = normalizerTransform
