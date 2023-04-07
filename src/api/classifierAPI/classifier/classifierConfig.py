from ..config import Config
import os
import json


class ClassifierConfig:

    def __init__(self, fileName) -> None:

        self._config = {
            'modelPath': 'model_latest.pth',
            'dataPath': 'first',
            'imageWidth': 62,
            'imageHeight': 62,
            'testRatio': 0.5,
            'saveModel': False,
            'loadModel': True,
            'dataLoaderWorkers': 2,
            'learningRate': 0.001,
            'epochs': 50,
            'momentum': 0.9
        }

        if fileName:
            self.setFromFile(fileName)
        self.overrideFromEnv()
        self.print()

    def setFromFile(self, fileName) -> None:
        try:
            with open(fileName, 'r') as configFile:
                configData = json.load(configFile)
                self.setFromJson(configData)
        except OSError:
            print(f'Error opening config file: {fileName}')

    def setFromJson(self, configData) -> None:
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

    def print(self) -> None:
        print(json.dumps(self._config, indent=4, sort_keys=True))

    def setModelPath(self, newValue) -> None:
        if newValue:
            self._config['modelPath'] = newValue

    def getModelPath(self) -> None:
        return os.path.join(Config.getModelsPath(), self._config['modelPath'])

    def setDataPath(self, newValue) -> None:
        if newValue:
            self._config['dataPath'] = newValue

    def getDataPath(self) -> None:
        return os.path.join(Config.getImagesPath(), self._config['dataPath'])

    def setImageWidth(self, newValue) -> None:
        if newValue:
            self._config['imageWidth'] = newValue

    def getImageWidth(self) -> None:
        return self.config['imageWidth']

    def setImageHeight(self, newValue) -> None:
        if newValue:
            self._config['imageHeight'] = newValue

    def getImageHeight(self) -> None:
        return self.config['imageHeight']

    def getImageSize(self) -> None:
        return (self.getImageWidth(), self.getImageHeight())

    def setTestRatio(self, newValue) -> None:
        if newValue:
            self._config['testRatio'] = newValue

    def getTestRatio(self) -> None:
        return self._config['testRatio']

    def setSaveModel(self, newValue) -> None:
        if newValue:
            self._config['saveModel'] = newValue

    def getSaveModel(self) -> None:
        return self._config['saveModel']

    def setLoadModel(self, newValue) -> None:
        if newValue:
            self._config['loadModel'] = newValue

    def getLoadModel(self) -> None:
        return self._config['loadModel']

    def setDataLoaderWorkers(self, newValue) -> None:
        if newValue:
            self._config['dataLoaderWorkers'] = newValue

    def getDataLoaderWorkers(self) -> None:
        return self._config['dataLoaderWorkers']

    def setLearningRate(self, newValue) -> None:
        if newValue:
            self._config['learningRate'] = newValue

    def getLearningRate(self) -> None:
        return self._config['learningRate']

    def setEpochs(self, newValue) -> None:
        if newValue:
            self._config['epochs'] = newValue

    def getEpochs(self) -> None:
        return self._config['epochs']

    def setMomentum(self, newValue) -> None:
        if newValue:
            self._config['momentum'] = newValue

    def getMomentum(self) -> None:
        return self._config['momentum']
