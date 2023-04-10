import torch
from .baseNetwork import BaseNetwork


class FirstNetwork(BaseNetwork):
    def __init__(self, numberOfClasses: int):
        super().__init__('first_network')
        layers = [
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=1,
                            padding=1, padding_mode="reflect"),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, kernel_size=3, stride=1,
                            padding=1, padding_mode="reflect"),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1,
                            padding=1, padding_mode="reflect"),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=1,
                            padding=1, padding_mode="reflect"),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1,
                            padding=1, padding_mode="reflect"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1,
                            padding=1, padding_mode="reflect"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(1),
            torch.nn.Linear(7200, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, numberOfClasses)
        ]

        self.layers = torch.nn.ModuleList(layers)
