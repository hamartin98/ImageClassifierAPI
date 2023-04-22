import cv2
import glob
import numpy as np
import os

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    '''Custom dataset to handle custom image data'''

    def __init__(self, path: str, classes: tuple, imgDim=(62, 62), transform=None):
        '''Default initalization'''

        self.transform = transform
        self.imagesPath = path
        fileList = glob.glob(os.path.join(self.imagesPath, "*"))

        self.data = []

        for classPath in fileList:
            classPath = os.path.relpath(classPath)
            className = os.path.normpath(classPath).split(os.path.sep)[-1]
            p = glob.glob(classPath + "/*.jpg")
            for imagePath in p:
                self.data.append([os.path.join(imagePath), className])

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

        if self.transform:
            imgTensor = self.transform(imgTensor)

        return imgTensor, classId

    def createClassMap(self, classes: tuple):
        '''Create class map'''

        classMap = {}
        for idx, cls in enumerate(classes):
            classMap[cls] = idx

        self.classMap = classMap
