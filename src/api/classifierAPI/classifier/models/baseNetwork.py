import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self, id: str):
        super().__init__()

        self.id: str = id
        self.layers: torch.nn.ModuleList = None

    def forward(self, x):
        result = x

        for layer in self.layers:
            result = layer(result)

        return result

    def getId(self) -> str:
        return self.id

    def load(self, path: str) -> None:
        try:
            self.load_state_dict(torch.load(path))
            print('Model loaded')
        except RuntimeError as error:
            print(f'Error loading model {error}')

    def loadToDevice(self, path: str, device) -> None:
        try:
            self.load_state_dict(torch.load(path, map_location=device))
            print('Model loaded')
        except RuntimeError as error:
            print(f'Error loading model: {error}')

    def save(self, path: str) -> None:
        try:
            torch.save(self.state_dict(), path)
            print('Model saved')
        except RuntimeError as error:
            print(f'Error saving model: {error}')
