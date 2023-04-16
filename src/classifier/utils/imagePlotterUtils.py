from typing import List
import numpy as np
from matplotlib import pyplot as plt


class ImagePlotterUtils:

    @staticmethod
    def plotLossAndAccuracy(lossData: List[float], accuracyData: List[float], savePath: str = 'lossAndAccuracy.png') -> None:
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
        y = np.arange(0, len(lossData))
        plt.title('Loss')
        plt.plot(y, lossData, label='Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig(savePath)
        plt.close()

    @staticmethod
    def plotAccuracyData(accuracyData: List[float], savePath: str = 'accuracy.png') -> None:
        y = np.arange(0, len(accuracyData))
        plt.title('Accuracy')
        plt.plot(y, accuracyData, label='Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig(savePath)
        plt.close()
