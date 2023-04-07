from typing import List
from classifierConfig import ClassifierConfig
from labelItem import LabelItem
from classificationType import ClassificationType


class BaseClassification:

    def __init__(self, name: str, type: ClassificationType, description: str) -> None:
        self.name = name
        self.type = type
        self.description = description
        self.classes = {}

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
    
    def configure(self, config) -> None:
        pass

    def classifyImageWithModel(image, config: ClassifierConfig) -> None:
        pass

    def classifyImagesWithModel(images, config: ClassifierConfig) -> None:
        pass


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

        self._classifcations = {
            'building': BuildingClassification(),
            'vegetation': VegetationClassification(),
            'road': PavedRoadClassification()
        }

        self._classifcationsByType = {
            ClassificationType.BUILDING: BuildingClassification(),
            ClassificationType.VEGETATION: VegetationClassification(),
            ClassificationType.ROAD: PavedRoadClassification()
        }

    def getClassificationByName(self, name) -> BaseClassification:
        if name in self._classifcations:
            return self._classifcations[name]

        return None

    def getClassificationByType(self, type: ClassificationType) -> BaseClassification:
        if type in self._classifcationsByType:
            return self._classifcations[type]

        return None
