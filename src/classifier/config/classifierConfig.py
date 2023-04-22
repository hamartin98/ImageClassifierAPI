import json
import os
from typing import Dict, Any, List, Tuple

from .config import Config
from ..classificationType import ClassificationType, ClassificationTypeUtils


class ClassifierConfig:
    '''Class to collect and manage configuration'''

    def __init__(self, fileName: str = None) -> None:
        '''Basic initialization'''

        self.originalJsonData = None

        self._config: Dict[str, Any] = {
            'modelPath': 'model_latest.pth',
            'dataPath': 'base_data/1000_b - SG',
            'imageWidth': 62,
            'imageHeight': 62,
            'trainRatio': 0.7,
            'testRatio': 0.1,
            'valRatio': 0.2,
            'saveModel': False,
            'loadModel': False,
            'dataLoaderWorkers': 2,
            'learningRate': 0.001,
            'epochs': 50,
            'momentum': 0.9,
            'batchSize': 10,
            'type': ClassificationType.NONE,
            'mean': [0.36720132, 0.38807531, 0.35384046],
            'std': [0.18385245, 0.17220756, 0.16941115]
        }

        if fileName:
            self.setFromFile(fileName)
        self.overrideFromEnv()
        # self.print()

    def setFromFile(self, fileName: str) -> None:
        '''Set config values from the given file'''

        try:
            with open(fileName, 'r') as configFile:
                configData = json.load(configFile)
                self.originalJsonData = configData
                self.setFromJson(configData)
        except OSError:
            print(f'Error opening config file: {fileName}')

    def setFromJson(self, configData: Dict[str, Any]) -> None:
        '''Set config values from the given json data'''

        if 'modelPath' in configData:
            self.setModelPath(configData['modelPath'])

        if 'dataPath' in configData:
            self.setDataPath(configData['dataPath'])

        if 'imageWidth' in configData:
            self.setImageWidth(configData['imageWidth'])

        if 'imageHeight' in configData:
            self.setImageHeight(configData['imageHeight'])

        if 'trainRatio' in configData:
            self.setTrainRatio(configData['trainRatio'])

        if 'testRatio' in configData:
            self.setTestRatio(configData['testRatio'])

        if 'valRatio' in configData:
            self.setValRatio(configData['valRatio'])

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

        if 'mean' in configData:
            self.setMean(configData['mean'])

        if 'std' in configData:
            self.setStd(configData['std'])

        self.originalJsonData = configData

    def overrideFromEnv(self) -> None:
        '''Override configuration values from environment'''

        self.setModelPath(os.environ.get('MODEL_PATH'))
        self.setDataPath(os.environ.get('DATA_PATH'))
        self.setImageWidth(os.environ.get('IMAGE_WIDTH'))
        self.setImageHeight(os.environ.get('IMAGE_HEIGHT'))
        self.setTrainRatio(os.environ.get('TRAIN_RATIO'))
        self.setTestRatio(os.environ.get('TEST_RATIO'))
        self.setValRatio(os.environ.get('VAL_RATIO'))
        self.setSaveModel(os.environ.get('SAVE_MODEL'))
        self.setLoadModel(os.environ.get('LOAD_MODEL'))
        self.setDataLoaderWorkers(os.environ.get('DATA_LOADER_WORKERS'))
        self.setLearningRate(os.environ.get('LEARNING_RATE'))
        self.setEpochs(os.environ.get('EPOCHS'))
        self.setMomentum(os.environ.get('MOMENTUM'))
        self.setBatchSize(os.environ.get('BATCH_SIZE'))
        # TODO: Set mean and std values

    def print(self) -> None:
        '''Print the configuration as a formatted json'''

        print(json.dumps(self._config, indent=4, sort_keys=True))

    def setModelPath(self, newValue: str) -> None:
        '''Set model path'''

        if Config.isSet(newValue):
            self._config['modelPath'] = newValue

    def getModelPath(self) -> str:
        '''Get model path'''

        path = os.path.join(Config.getModelsPath(), self._config['modelPath'])
        if Config.getIsRelativePath():
            return os.path.realpath(path)
        return os.path.abspath(path)

    def setDataPath(self, newValue: str) -> None:
        '''Set training data path'''

        if Config.isSet(newValue):
            self._config['dataPath'] = newValue

    def getDataPath(self) -> str:
        '''Get training data path'''

        path = os.path.join(Config.getImagesPath(), self._config['dataPath'])
        if Config.getIsRelativePath():
            return os.path.realpath(path)
        return os.path.abspath(path)

    def setImageWidth(self, newValue: int) -> None:
        '''Set image width'''

        if Config.isSet(newValue):
            self._config['imageWidth'] = newValue

    def getImageWidth(self) -> int:
        '''Get image width'''

        return self._config['imageWidth']

    def setImageHeight(self, newValue: int) -> None:
        '''Set image height'''

        if Config.isSet(newValue):
            self._config['imageHeight'] = newValue

    def getImageHeight(self) -> int:
        '''Get image height'''

        return self._config['imageHeight']

    def getImageSize(self) -> Tuple:
        '''Get image size as a tuple in (width, height) format'''

        return (self.getImageWidth(), self.getImageHeight())

    def setTrainRatio(self, newValue: float) -> None:
        '''Set training dataset ratio'''

        if Config.isSet(newValue):
            self._config['trainRatio'] = newValue

    def getTrainRatio(self) -> float:
        '''Get training dataset ratio'''

        return self._config['trainRatio']

    def setTestRatio(self, newValue: float) -> None:
        '''Set test dataset ratio'''

        if Config.isSet(newValue):
            self._config['testRatio'] = newValue

    def getTestRatio(self) -> float:
        '''Get test dataset ratio'''

        return self._config['testRatio']

    def setValRatio(self, newValue: float) -> None:
        '''Set validation dateset ratio'''

        if Config.isSet(newValue):
            self._config['valRatio'] = newValue

    def getValRatio(self) -> None:
        '''Get validation dataset ratio'''

        return self._config['valRatio']

    def setSaveModel(self, newValue: bool) -> None:
        '''Set path to save model to'''

        if Config.isSet(newValue):
            self._config['saveModel'] = newValue

    def getSaveModel(self) -> bool:
        '''Get model save path'''

        return self._config['saveModel']

    def setLoadModel(self, newValue: bool) -> None:
        '''Set whether to load an existing model'''

        if Config.isSet(newValue):
            self._config['loadModel'] = newValue

    def getLoadModel(self) -> bool:
        '''Get load model status'''

        return self._config['loadModel']

    def setDataLoaderWorkers(self, newValue: int) -> None:
        '''Set the number of dataloader workers'''

        if Config.isSet(newValue):
            self._config['dataLoaderWorkers'] = newValue

    def getDataLoaderWorkers(self) -> int:
        '''Get the number of dataloader workers'''

        return self._config['dataLoaderWorkers']

    def setLearningRate(self, newValue: float) -> None:
        '''Set learning rate'''

        if Config.isSet(newValue):
            self._config['learningRate'] = newValue

    def getLearningRate(self) -> float:
        '''Get learning rate'''

        return self._config['learningRate']

    def setEpochs(self, newValue: int) -> None:
        '''Set number of epochs'''

        if Config.isSet(newValue):
            self._config['epochs'] = newValue

    def getEpochs(self) -> int:
        '''Get number of epochs'''

        return self._config['epochs']

    def setMomentum(self, newValue: float) -> None:
        '''Set momentum'''

        if Config.isSet(newValue):
            self._config['momentum'] = newValue

    def getMomentum(self) -> float:
        '''Get momentum'''

        return self._config['momentum']

    def setBatchSize(self, newValue) -> int:
        '''Set number of batches'''

        if Config.isSet(newValue):
            self._config['batchSize'] = newValue

    def getBatchSize(self) -> int:
        '''Get number of batches'''

        return self._config['batchSize']

    def setType(self, newValue) -> None:
        '''Set classification type'''

        if Config.isSet(newValue):
            if isinstance(newValue, str):
                self._config['type'] = ClassificationTypeUtils.fromString(
                    newValue)
            elif isinstance(newValue, ClassificationType):
                self._config['type'] = newValue
            else:
                self._config['type'] = ClassificationType.NONE

    def innerOverrideToType(self, type: ClassificationType) -> None:
        '''Override current config to the given type'''

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
        '''Override the current configuration with the given value according to the given type'''

        self.setType(type)
        self.setFromJson(data)

    def getType(self) -> ClassificationType:
        '''Get current classification type'''

        return self._config['type']

    def setMean(self, newValue: List[float]) -> None:
        '''Set mean values'''

        if isinstance(newValue, list) and len(newValue) == 3:
            self._config['mean'] = newValue
        else:
            raise ValueError('Invalid mean value in configuration')

    def getMean(self) -> List[int]:
        '''Get mean values'''

        return self._config['mean']

    def setStd(self, newValue: List[float]) -> None:
        '''Set std values'''

        if isinstance(newValue, list) and len(newValue) == 3:
            self._config['std'] = newValue
        else:
            raise ValueError('Invalid std value in configuration')

    def getStd(self) -> List[int]:
        '''Get std values'''

        return self._config['std']

    def getAsJson(self) -> Dict[str, Any]:
        '''Return configuration as a json dictionary'''

        asJson = self._config
        type = asJson['type']
        asJson['type'] = str(type)
        return asJson
