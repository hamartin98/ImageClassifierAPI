import os
from config import Config
from classifierConfig import ClassifierConfig

from teacher import Teacher
from classificationMap import BuildingClassification
from multiModelTeacher import MultiModelTeacher

multiMode = True


def teachMultipleModels(config: ClassifierConfig) -> None:
    teacher = MultiModelTeacher(config)
    teacher.teachWithClassifiers()


if __name__ == '__main__':
    currDir = os.path.abspath(os.getcwd())  # Working directory
    path = os.path.normpath(os.path.join(
        currDir, 'src/' 'config.json'))
    Config(path)
    
    config = ClassifierConfig(Config.getPath())
    
    if not multiMode:
        '''currDir = os.path.abspath(os.getcwd())  # Working directory
        path = os.path.normpath(os.path.join(
            currDir, 'src/' 'config.json'))
        Config(path)'''

        classification = BuildingClassification()

        teacher = Teacher(classification, config)
        config.print()
        teacher.train()
    else:
        teachMultipleModels(config)
