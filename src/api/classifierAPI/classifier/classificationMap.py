class LabelItem:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def getName(self) -> None:
        return self.name

    def getDescription(self) -> None:
        return self.description


class BaseClassification:

    def __init__(self, description) -> None:
        self.description = description
        self.classes = {}

    def getLabelByValue(self, value) -> None:
        if value in self.classes:
            return self.classes[value]

        return None

    def getDescription(self) -> None:
        return self.description


class BuildingClassification(BaseClassification):
    def __init__(self) -> None:
        super().__init__('Building ratio classification, represents ratio of buildings')
        self.classes = {
            0: LabelItem(0, 'Ratio of buildings is 0%'),
            1: LabelItem(1, 'Ratio of buildings is between 0 and 50%'),
            2: LabelItem(2, 'Ratio of buildings is above 50%')
        }


class VegetationClassification(BaseClassification):
    def __init__(self) -> None:
        super().__init__('Vegetation ratio classification, represents ratio of vegetation')
        self.classes = {
            0: LabelItem(0, 'Ratio of vegetation is 0%'),
            1: LabelItem(1, 'Ratio of vegetation is between 0 and 50%'),
            2: LabelItem(2, 'Ratio of vegetation is above 50%')
        }


class PavedRoadClassification(BaseClassification):
    def __init__(self) -> None:
        super().__init__('Paved road classification, represents if paved road is present or not')
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

    def getClassificationByName(self, name) -> BaseClassification:
        if name in self._classifcations:
            return self._classifcations[name]

        return None
