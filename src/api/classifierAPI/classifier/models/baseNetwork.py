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
