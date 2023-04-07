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
        self.load_state_dict(torch.load(path))
        print('Model loaded')

    def save(self, path: str) -> None:
        self._save_to_state_dict()
        torch.save(self.state_dict(), path)
        print('Model saved')
