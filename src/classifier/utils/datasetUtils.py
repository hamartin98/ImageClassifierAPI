from sklearn.model_selection import train_test_split
from typing import Dict

import torch
from torch.utils.data import Dataset, Subset, random_split

'''Collection of dataset related utility functions'''


def splitDatasetTrainTest(dataSet: Dataset, testSize: float) -> Dict[str, Subset]:
    '''Split dataset into training and test sets'''
    dataSets = {}
    arrays = list(range(len(dataSet)))
    trainIdx, testIdx = train_test_split(arrays, test_size=testSize)

    # set corresponding data in the dictionary
    dataSets['train'] = Subset(dataSet, trainIdx)
    dataSets['test'] = Subset(dataSet, testIdx)

    return dataSets


def splitDataSet(dataSet: Dataset, trainRatio: float, valRatio: float, testRatio: float) -> Dict[str, Subset]:
    '''Split dataset into training, test and validation sets with the given ratios'''
    generator = torch.Generator()
    ratios = [trainRatio, testRatio, valRatio]
    trainIdx, testIdx, valIdx = random_split(dataSet, ratios, generator)

    dataSets = {}

    dataSets['train'] = Subset(dataSet, trainIdx.indices)
    dataSets['test'] = Subset(dataSet, testIdx.indices)
    dataSets['val'] = Subset(dataSet, valIdx.indices)

    return dataSets