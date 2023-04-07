from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def splitDataset(dataSet, testSize):
    '''Split dataset into training and test sets'''
    dataSets = {}
    arrays = list(range(len(dataSet)))
    trainIdx, testIdx = train_test_split(arrays, test_size=testSize)

    # set corresponding data in the dictionary
    dataSets['train'] = Subset(dataSet, trainIdx)
    dataSets['test'] = Subset(dataSet, testIdx)

    return dataSets
