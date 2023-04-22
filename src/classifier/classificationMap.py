import os
from typing import List, Dict, Tuple

from .classificationType import ClassificationType
from .config.classifierConfig import ClassifierConfig
from .config.config import Config
from .labelItem import LabelItem
from .models.baseNetwork import BaseNetwork
from .models.FirstNetwork import FirstNetwork


class BaseClassification:
    '''Basic classification information class'''

    def __init__(self, name: str, type: ClassificationType, description: str) -> None:
        '''Init classification information'''

        self.name = name
        self.type = type
        self.description = description
        self.classes = {}
        self.configuration: ClassifierConfig = None
        self.network: BaseNetwork = None
        self.device: str = None

    def getLabelByValue(self, value) -> LabelItem:
        '''Return label item by value'''

        if value in self.classes:
            return self.classes[value]

        return None

    def getName(self) -> str:
        '''Get classification's name'''

        return self.name

    def getType(self) -> ClassificationType:
        '''Get classifications's type'''

        return self.type

    def getDescription(self) -> str:
        '''Get classifications's description'''

        return self.description

    def getClassLabels(self) -> List[int]:
        '''Get list of labels associated with the classification'''

        labelList = list(str(key) for key in self.classes.keys())
        return labelList

    def getClassLabelsTuple(self) -> Tuple:
        '''Return the list of labels as a tuple'''

        return tuple(self.getClassLabels())

    def getClassNum(self) -> int:
        '''Get the number of current classes'''

        return len(self.classes)

    def configure(self, config: ClassifierConfig) -> None:
        '''Configure the current classification'''

        self.configuration = config
        if self.configuration.getType() != self.type:
            self.configuration.innerOverrideToType(self.type)

    def setupNetwork(self, device: str) -> None:
        '''Setup classification's network'''

        self.device = device
        self.network = FirstNetwork(self.getClassNum())
        if self.configuration.getLoadModel():
            self.network.loadToDevice(
                self.configuration.getModelPath(), self.device)

    def configureAndSetupNetwork(self, config: ClassifierConfig) -> None:
        '''Configure classification and setup network'''

        self.configure(config)
        self.setupNetwork(self.device)

    def isConfigured(self) -> bool:
        '''Return whether the classification is configured'''

        return self.configuration is not None and self.configuration.getType() == self.type

    def getConfigutation(self) -> ClassifierConfig:
        '''Return current classifications configuration'''

        return self.configuration

    def getNetwork(self) -> BaseNetwork:
        '''Get current classifications network'''

        return self.network

    def classifyImageWithModel(image, config: ClassifierConfig) -> None:
        '''Classify the given image with the current classification'''

        # TODO: implement
        pass

    def classifyImagesWithModel(images, config: ClassifierConfig) -> None:
        '''Classify multiple images with the current classification'''

        # TODO: implement
        pass

    # TODO: Fix path issues
    def saveModel(self) -> None:
        '''Save current classification's model'''

        if self.configuration.getSaveModel():
            if self.network:
                basePath = Config.getModelsPath()
                #modelName = self.network.getId() + '_' + self.name + '_2.pth'
                modelName = self.configuration.getModelPath()
                basePath = 'data\models'
                #savePath = os.path.normpath(os.path.abspath(os.path.join(basePath, modelName)))
                savePath = os.path.join(basePath, modelName)
                savePath = os.path.normpath(savePath)
                self.network.save(savePath)
            else:
                print('Error saving model, network not found')

    def loadModel(self) -> None:
        '''Load current classification's model'''

        print('TODO: implement model load')


class BuildingClassification(BaseClassification):
    '''Building classification information'''

    def __init__(self) -> None:
        super().__init__('building', ClassificationType.BUILDING,
                         'Building ratio classification, represents ratio of buildings')
        self.classes = {
            0: LabelItem(0, 'Ratio of buildings is 0%'),
            1: LabelItem(1, 'Ratio of buildings is between 0 and 50%'),
            2: LabelItem(2, 'Ratio of buildings is above 50%')
        }


class VegetationClassification(BaseClassification):
    '''Vegetation classificatio information'''

    def __init__(self) -> None:
        super().__init__('vegetation',
                         ClassificationType.VEGETATION,
                         'Vegetation ratio classification, represents ratio of vegetation')
        self.classes = {
            0: LabelItem(0, 'Ratio of vegetation is 0%'),
            1: LabelItem(1, 'Ratio of vegetation is between 0 and 50%'),
            2: LabelItem(2, 'Ratio of vegetation is above 50%')
        }


class PavedRoadClassification(BaseClassification):
    '''Road classification information'''

    def __init__(self) -> None:
        super().__init__(
            'road', ClassificationType.ROAD, 'Paved road classification, represents if paved road is present or not')
        self.classes = {
            0: LabelItem(0, 'No paved road is present'),
            1: LabelItem(1, 'Paved road is present')
        }


class ClassificationMap:
    '''Map classifition types to classifications'''

    def __init__(self) -> None:
        '''Initialite classifications map'''

        self._classifcations: Dict[str, BaseClassification] = {
            'building': BuildingClassification(),
            'vegetation': VegetationClassification(),
            'road': PavedRoadClassification()
        }

        self._classifcationsByType: Dict[ClassificationType, BaseClassification] = {
            ClassificationType.BUILDING: BuildingClassification(),
            ClassificationType.VEGETATION: VegetationClassification(),
            ClassificationType.ROAD: PavedRoadClassification()
        }

    def getClassifications(self) -> Dict[str, BaseClassification]:
        '''Get classifications map'''

        return self._classifcations

    def getClassificationsByType(self) -> None:
        '''Return classification map by type'''

        return self._classifcationsByType

    def getClassificationByName(self, name: str) -> BaseClassification:
        '''Return classification associated with the given name'''

        if name in self._classifcations:
            return self._classifcations[name]

        return None

    def getClassificationByType(self, type: ClassificationType) -> BaseClassification:
        '''Return classification associated with the given classification type'''

        if type in self._classifcationsByType:
            return self._classifcationsByType[type]

        return None
