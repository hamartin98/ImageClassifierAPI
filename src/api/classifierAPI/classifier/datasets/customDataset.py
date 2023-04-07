import os
import glob
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, path, classes, imgDim=(32, 32), transform=None):
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
        return len(self.data)

    def __getitem__(self, idx):
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

    def createClassMap(self, classes):
        classMap = {}
        for idx, cls in enumerate(classes):
            classMap[cls] = idx

        self.classMap = classMap
