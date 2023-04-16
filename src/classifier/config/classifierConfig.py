from typing import Dict, Any

from .config import Config
import os
import json
from ..classificationType import ClassificationType, ClassificationTypeUtils


class ClassifierConfig:

    def __init__(self, fileName) -> None:

        self.originalJsonData = None

        self._config: Dict[str, Any] = {
            'modelPath': 'model_latest.pth',
            'dataPath': 'base_data/1000_b - SG',
            'imageWidth': 62,
            'imageHeight': 62,
            'testRatio': 0.5,
            'saveModel': False,
            'loadModel': True,
            'dataLoaderWorkers': 2,
            'learningRate': 0.001,
            'epochs': 50,
            'momentum': 0.9,
            'batchSize': 10,
            'type': ClassificationType.NONE
        }

        if fileName:
            self.setFromFile(fileName)
        self.overrideFromEnv()
        self.print()

    def setFromFile(self, fileName: str) -> None:
        try:
            with open(fileName, 'r') as configFile:
                configData = json.load(configFile)
                self.originalJsonData = configData
                self.setFromJson(configData)
        except OSError:
            print(f'Error opening config file: {fileName}')

    def setFromJson(self, configData: Dict[str, Any]) -> None:
        if 'modelPath' in configData:
            self.setModelPath(configData['modelPath'])

        if 'dataPath' in configData:
            self.setDataPath(configData['dataPath'])

        if 'imageWidth' in configData:
            self.setImageWidth(configData['imageWidth'])

        if 'imageHeight' in configData:
            self.setImageHeight(configData['imageHeight'])

        if 'testRatio' in configData:
            self.setTestRatio(configData['testRatio'])

        if 'saveModel' in configData:
            self.setSaveModel(configData['saveModel'])

        if 'loadModel' in configData:
            self.setLoadModel(configData['loadModel'])

        if 'dataLoaderWorkers' in configData:
            self.setDataLoaderWorkers(configData['dataLoaderWorkers'])

        if 'learningRate' in configData:
            self.setLearningRate(configData['learningRate'])

        if 'epochs' in configData:
            self.setEpochs(configData['epochs'])

        if 'momentum' in configData:
            self.setMomentum(configData['momentum'])

        if 'batchSize' in configData:
            self.setBatchSize(configData['batchSize'])

        if 'type' in configData:
            self.setType(configData['type'])

        self.originalJsonData = configData

    def overrideFromEnv(self) -> None:
        self.setModelPath(os.environ.get('MODEL_PATH'))
        self.setDataPath(os.environ.get('DATA_PATH'))
        self.setImageWidth(os.environ.get('IMAGE_WIDTH'))
        self.setImageHeight(os.environ.get('IMAGE_HEIGHT'))
        self.setTestRatio(os.environ.get('TEST_RATIO'))
        self.setSaveModel(os.environ.get('SAVE_MODEL'))
        self.setLoadModel(os.environ.get('LOAD_MODEL'))
        self.setDataLoaderWorkers(os.environ.get('DATA_LOADER_WORKERS'))
        self.setLearningRate(os.environ.get('LEARNING_RATE'))
        self.setEpochs(os.environ.get('EPOCHS'))
        self.setMomentum(os.environ.get('MOMENTUM'))
        self.setBatchSize(os.environ.get('BATCH_SIZE'))

    def print(self) -> None:
        print(json.dumps(self._config, indent=4, sort_keys=True))

    def setModelPath(self, newValue: str) -> None:
        if Config.isSet(newValue):
            self._config['modelPath'] = newValue

    def getModelPath(self) -> str:
        path = os.path.join(Config.getModelsPath(), self._config['modelPath'])
        if Config.getIsRelativePath():
            return os.path.realpath(path)
        return os.path.abspath(path)

    def setDataPath(self, newValue: str) -> None:
        if Config.isSet(newValue):
            self._config['dataPath'] = newValue

    def getDataPath(self) -> str:
        path = os.path.join(Config.getImagesPath(), self._config['dataPath'])
        if Config.getIsRelativePath():
            return os.path.realpath(path)
        return os.path.abspath(path)

    def setImageWidth(self, newValue: int) -> None:
        if Config.isSet(newValue):
            self._config['imageWidth'] = newValue

    def getImageWidth(self) -> int:
        return self._config['imageWidth']

    def setImageHeight(self, newValue: int) -> None:
        if Config.isSet(newValue):
            self._config['imageHeight'] = newValue

    def getImageHeight(self) -> int:
        return self._config['imageHeight']

    def getImageSize(self) -> tuple:
        return (self.getImageWidth(), self.getImageHeight())

    def setTestRatio(self, newValue: float) -> None:
        if Config.isSet(newValue):
            self._config['testRatio'] = newValue

    def getTestRatio(self) -> float:
        return self._config['testRatio']

    def setSaveModel(self, newValue: bool) -> None:
        if Config.isSet(newValue):
            self._config['saveModel'] = newValue

    def getSaveModel(self) -> bool:
        return self._config['saveModel']

    def setLoadModel(self, newValue: bool) -> None:
        if Config.isSet(newValue):
            self._config['loadModel'] = newValue

    def getLoadModel(self) -> bool:
        return self._config['loadModel']

    def setDataLoaderWorkers(self, newValue: int) -> None:
        if Config.isSet(newValue):
            self._config['dataLoaderWorkers'] = newValue

    def getDataLoaderWorkers(self) -> int:
        return self._config['dataLoaderWorkers']

    def setLearningRate(self, newValue: float) -> None:
        if Config.isSet(newValue):
            self._config['learningRate'] = newValue

    def getLearningRate(self) -> float:
        return self._config['learningRate']

    def setEpochs(self, newValue: int) -> None:
        if Config.isSet(newValue):
            self._config['epochs'] = newValue

    def getEpochs(self) -> int:
        return self._config['epochs']

    def setMomentum(self, newValue: float) -> None:
        if Config.isSet(newValue):
            self._config['momentum'] = newValue

    def getMomentum(self) -> float:
        return self._config['momentum']

    def setBatchSize(self, newValue) -> int:
        if Config.isSet(newValue):
            self._config['batchSize'] = newValue

    def getBatchSize(self) -> int:
        return self._config['batchSize']

    def setType(self, newValue) -> None:
        if Config.isSet(newValue):
            if isinstance(newValue, str):
                self._config['type'] = ClassificationTypeUtils.fromString(
                    newValue)
            elif isinstance(newValue, ClassificationType):
                self._config['type'] = newValue
            else:
                self._config['type'] = ClassificationType.NONE

    def innerOverrideToType(self, type: ClassificationType) -> None:
        data = None
        if type == ClassificationType.BUILDING:
            if 'building' in self.originalJsonData:
                data = self.originalJsonData['building']
        elif type == ClassificationType.VEGETATION:
            if 'vegetation' in self.originalJsonData:
                data = self.originalJsonData['vegetation']
        elif type == ClassificationType.ROAD:
            if 'road' in self.originalJsonData:
                data = self.originalJsonData['road']

        if data:
            self.overrideToType(data, type)

    def overrideToType(self, data, type: ClassificationType) -> None:
        self.setType(type)
        self.setFromJson(data)

    def getType(self) -> ClassificationType:
        return self._config['type']

    def getAsJson(self) -> Dict[str, Any]:
        asJson = self._config
        type = asJson['type']
        asJson['type'] = str(type)
        return asJson
