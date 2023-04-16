from typing import Dict, Any
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, random_split
import torch


def splitDatasetOld(dataSet, testSize):
    '''Split dataset into training and test sets'''
    dataSets = {}
    arrays = list(range(len(dataSet)))
    trainIdx, testIdx = train_test_split(arrays, test_size=testSize)

    # set corresponding data in the dictionary
    dataSets['train'] = Subset(dataSet, trainIdx)
    dataSets['test'] = Subset(dataSet, testIdx)

    return dataSets


def splitDataSet(dataSet, trainRatio: float, valRatio: float, testRatio: float) -> Dict[str, Any]:
    generator = torch.Generator()
    ratios = [trainRatio, testRatio, valRatio]
    trainIdx, testIdx, valIdx = random_split(dataSet, ratios, generator)

    dataSets = {}

    dataSets['train'] = Subset(dataSet, trainIdx.indices)
    dataSets['test'] = Subset(dataSet, testIdx.indices)
    dataSets['val'] = Subset(dataSet, valIdx.indices)
    
    return dataSets
