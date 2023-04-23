import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    '''Basic network class to build new models on top of it'''

    def __init__(self, id: str):
        '''Basic initialization'''

        super().__init__()

        self.id: str = id
        self.layers: torch.nn.ModuleList = None

    def forward(self, x):
        '''Override forward function'''

        result = x

        for layer in self.layers:
            result = layer(result)

        return result

    def getId(self) -> str:
        '''Get the id of the current network'''

        return self.id

    def load(self, path: str) -> None:
        '''Load model state from the given file'''

        try:
            self.load_state_dict(torch.load(path))
            print('Model loaded')
        except RuntimeError as error:
            print(f'Error loading model {error}')

    def loadToDevice(self, path: str, device) -> None:
        '''Load model state to the given device'''

        try:
            self.load_state_dict(torch.load(path, map_location=device))
            print('Model loaded')
        except RuntimeError as error:
            print(f'Error loading model: {error}')

    def save(self, path: str) -> None:
        '''Save the current model to the given path'''

        try:
            print(path)
            torch.save(self.state_dict(), path)
            print('Model saved')
        except RuntimeError as error:
            print(f'Error saving model: {error}')
