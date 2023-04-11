from typing import List, Dict
import os
from .classifierConfig import ClassifierConfig
from .config import Config
from .labelItem import LabelItem
from .classificationType import ClassificationType
from .models.baseNetwork import BaseNetwork
from .models.FirstNetwork import FirstNetwork


class BaseClassification:

    def __init__(self, name: str, type: ClassificationType, description: str) -> None:
        self.name = name
        self.type = type
        self.description = description
        self.classes = {}
        self.configuration: ClassifierConfig = None
        self.network: BaseNetwork = None
        self.device: str = None

    def getLabelByValue(self, value) -> LabelItem:
        if value in self.classes:
            return self.classes[value]

        return None

    def getName(self) -> str:
        return self.name

    def getType(self) -> ClassificationType:
        return self.type

    def getDescription(self) -> str:
        return self.description

    def getClassLabels(self) -> List[int]:
        labelList = list(str(key) for key in self.classes.keys())
        return labelList

    def getClassLabelsTuple(self) -> tuple:
        return tuple(self.getClassLabels())

    def getClassNum(self) -> int:
        return len(self.classes)

    def configure(self, config: ClassifierConfig) -> None:
        self.configuration = config
        if self.configuration.getType() != self.type:
            self.configuration.innerOverrideToType(self.type)

    def setupNetwork(self, device: str) -> None:
        self.device = device
        self.network = FirstNetwork(self.getClassNum())
        if self.configuration.getLoadModel():
            self.network.loadToDevice(
                self.configuration.getModelPath(), self.device)

    def configureAndSetupNetwork(self, config: ClassifierConfig) -> None:
        self.configure(config)
        self.setupNetwork(self.device)

    def isConfigured(self) -> bool:
        return self.configuration is not None and self.configuration.getType() == self.type

    def getConfigutation(self) -> ClassifierConfig:
        return self.configuration

    def getNetwork(self) -> BaseNetwork:
        return self.network

    def classifyImageWithModel(image, config: ClassifierConfig) -> None:
        pass

    def classifyImagesWithModel(images, config: ClassifierConfig) -> None:
        pass

    # TODO: Fix path issues
    def saveModel(self) -> None:
        if self.configuration.getSaveModel():
            if self.network:
                basePath = Config.getModelsPath()
                modelName = self.network.getId() + '_' + self.name + '_2.pth'
                basePath = 'data\models'
                #savePath = os.path.normpath(os.path.abspath(os.path.join(basePath, modelName)))
                savePath = os.path.join(basePath, modelName)
                savePath = os.path.normpath(savePath)
                self.network.save(savePath)
            else:
                print('Error saving model, network not found')

    def loadModel(self) -> None:
        print('TODO: implement model load')


class BuildingClassification(BaseClassification):
    def __init__(self) -> None:
        super().__init__('building', ClassificationType.BUILDING,
                         'Building ratio classification, represents ratio of buildings')
        self.classes = {
            0: LabelItem(0, 'Ratio of buildings is 0%'),
            1: LabelItem(1, 'Ratio of buildings is between 0 and 50%'),
            2: LabelItem(2, 'Ratio of buildings is above 50%')
        }


class VegetationClassification(BaseClassification):
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
    def __init__(self) -> None:
        super().__init__(
            'road', ClassificationType.ROAD, 'Paved road classification, represents if paved road is present or not')
        self.classes = {
            0: LabelItem(0, 'No paved road is present'),
            1: LabelItem(1, 'Paved road is present')
        }


class ClassificationMap:

    def __init__(self) -> None:

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

    def getClassifications(self) -> None:
        return self._classifcations

    def getClassificationsByType(self) -> None:
        return self._classifcationsByType

    def getClassificationByName(self, name) -> BaseClassification:
        if name in self._classifcations:
            return self._classifcations[name]

        return None

    def getClassificationByType(self, type: ClassificationType) -> BaseClassification:
        if type in self._classifcationsByType:
            return self._classifcationsByType[type]

        return None
