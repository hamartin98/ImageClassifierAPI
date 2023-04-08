from typing import Dict
from classifierConfig import ClassifierConfig
from classificationMap import (
    BaseClassification)

from classificationMap import ClassificationMap, BaseClassification
from teacher import Teacher


class MultiModelTeacher:
    def __init__(self, config: ClassifierConfig) -> None:
        self.baseConfig = config
        self.classifications = ClassificationMap()

        self.setupClassifiers()

    def setupClassifiers(self) -> None:
        classifications: Dict[str,
                              BaseClassification] = self.classifications.getClassifications()

        for key, classification in classifications.items():
            classification.configureAndSetupNetwork(self.baseConfig)

    def teachWithClassifiers(self) -> None:
        classifications: Dict[str,
                              BaseClassification] = self.classifications.getClassifications()

        for key, classification in classifications.items():
            print(key)
            if key != 'building':
                classification.configureAndSetupNetwork(self.baseConfig)
                classification.getConfigutation().print()
                teacher = Teacher(classification)
                teacher.trainAndTest()
