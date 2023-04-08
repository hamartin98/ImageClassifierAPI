import os
from config import Config
from classifierConfig import ClassifierConfig

from teacher import Teacher
from classificationMap import BuildingClassification
from multiModelClassifier import MultiModelClassifier
from classificationType import ClassificationType

if __name__ == '__main__':
    currDir = os.path.abspath(os.getcwd()) # Working directory
    path = os.path.normpath(os.path.join(currDir, 'src/api/classifierAPI/' 'config.json'))
    Config(path)
    
    classifier = MultiModelClassifier()
    
    classification = BuildingClassification()
    config = ClassifierConfig(Config.getPath())
    
    teacher = Teacher(classification, config)
    config.print()
    teacher.train()