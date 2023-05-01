from typing import List, Any
import random

import torchvision.transforms as T
import torchvision.transforms.functional as F


class RotationTransform:
    '''Custom rotation transform class to apply only 90 degree rotations'''

    def __call__(self, x):
        '''Override call'''

        angle = random.choice([0, 90, 180, 270])
        return F.rotate(x, angle)


class Transformations():
    '''Collection of dataset transformations'''

    def __init__(self) -> None:
        '''Init'''
        
        self.basicAugmenterTransforms = []
        
        self.initBasicAugmenterTransforms()

    def initBasicAugmenterTransforms(self) -> None:
        '''Setup basic augmenter transformation'''
        
        toTensor = T.ToTensor()
        randomSharpness = T.RandomAdjustSharpness(sharpness_factor=20)
        gaussianBlur = T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 5))
        colorJitter = T.ColorJitter(
            brightness=0.5, contrast=0.3, saturation=0.5)
        randomHFlip = T.RandomHorizontalFlip(p=0.5)
        randomVFlip = T.RandomVerticalFlip(p=0.5)
        randomRotation = RotationTransform()

        self.basicAugmenterTransforms = [
            toTensor, randomSharpness, gaussianBlur, colorJitter, randomHFlip, randomVFlip, randomRotation]

    def getBasicAugmenterTransforms(self) -> List[Any]:
        '''Get basic data augmenter transformation list'''

        return self.basicAugmenterTransforms
