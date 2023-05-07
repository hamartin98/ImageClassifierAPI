import torch
import torch.nn as nn
from torchvision import models


class ResNetNetwork:
    '''Wrapper class for resnet to use like custom model'''

    def __init__(self, id: str = 'resnet50'):
        self.id = id
        self.model = models.resnet50(weights=None)

    def getId(self) -> str:
        '''Get network's id'''

        return self.id

    def getModel(self) -> nn.Module:
        '''Get network's model'''

        return self.model

    def load(self, path: str) -> None:
        '''Load model state from the given file'''

        try:
            self.model.load_state_dict(torch.load(path))
            print('Model loaded')
        except RuntimeError as error:
            print(f'Error loading model {error}')

    def loadToDevice(self, path: str, device: str) -> None:
        '''Load model state to the given device'''

        try:
            self.model.load_state_dict(torch.load(path, map_location=device))
            print('Model loaded')
        except RuntimeError as error:
            print(f'Error loading model: {error}')

    def save(self, path: str) -> None:
        '''Save the current model to the given path'''

        try:
            print(path)
            torch.save(self.model.state_dict(), path)
            print('Model saved')
        except RuntimeError as error:
            print(f'Error saving model: {error}')
