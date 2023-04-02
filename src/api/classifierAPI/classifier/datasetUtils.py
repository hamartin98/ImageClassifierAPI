from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def splitDataset(dataset, test_size):
    '''Split dataset into training and test sets'''
    datasets = {}
    arrays = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(arrays, test_size=test_size)

    # set corresponding data in the dictionary
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, test_idx)

    return datasets
