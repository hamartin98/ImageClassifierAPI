from matplotlib import pyplot as plt
import numpy as np
from typing import List


class ImagePlotterUtils:
    '''Collection of image plotting related utility functions'''

    @staticmethod
    def plotLossAndAccuracy(lossData: List[float], accuracyData: List[float], savePath: str = 'lossAndAccuracy.png') -> None:
        '''Plot loss and accuracy values and save the result to the given path'''

        y = np.arange(0, max(len(lossData), len(accuracyData)))
        plt.title('Loss and accuracy')
        plt.plot(y, lossData, label='Loss')
        plt.plot(y, accuracyData, label='Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig(savePath)
        plt.close()

    @staticmethod
    def plotLossData(lossData: List[float], savePath: str = 'loss.png') -> None:
        '''Plot loss data and save result to the given path'''

        y = np.arange(0, len(lossData))
        plt.title('Loss')
        plt.plot(y, lossData, label='Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig(savePath)
        plt.close()

    @staticmethod
    def plotAccuracyData(accuracyData: List[float], savePath: str = 'accuracy.png') -> None:
        '''Plot accuracy values and save the result to the given path'''

        y = np.arange(0, len(accuracyData))
        plt.title('Accuracy')
        plt.plot(y, accuracyData, label='Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig(savePath)
        plt.close()
