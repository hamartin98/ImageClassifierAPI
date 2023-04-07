import json
import os


class Config:
    _inited = False
    _config = {
            'basePath': 'data',
            'isRelativePath': True
        }

    def __init__(self, fileName) -> None:
        
        if not Config._inited:

            Config.setFromFile(fileName)
            Config.overrideFromEnv()
            Config.print()
            Config._inited = True

    @staticmethod
    def setFromFile(fileName) -> None:
        try:
            with open(fileName, 'r') as configFile:
                configData = json.load(configFile)
                if 'basePath' in configData:
                    Config.setBasePath(configData['basePath'])
                if 'isRelativePath' in configData:
                    Config.setIsRelativePath(configData['isRelativePath'])
        except OSError:
            print(f'Error opening config file: {fileName}')

    @staticmethod
    def overrideFromEnv() -> None:
        Config.setBasePath(os.environ.get('BASE_PATH'))
        Config.setIsRelativePath(os.environ.get('IS_RELATIVE_PATH'))

    @staticmethod
    def getBasePath() -> None:
        return Config._config['basePath']

    @staticmethod
    def setBasePath(newValue) -> None:
        if newValue:
            Config._config['basePath'] = newValue

    @staticmethod
    def getIsRelativePath() -> None:
        return Config._config['isRelativePath']

    @staticmethod
    def setIsRelativePath(newValue) -> None:
        if newValue:
            Config._config['isRelativePath'] = newValue

    @staticmethod
    def getImagesPath() -> None:
        return os.path.join(Config._config['basePath'], 'images')

    @staticmethod
    def getModelsPath() -> None:
        return os.path.join(Config._config['basePath'], 'models')

    @staticmethod
    def print() -> None:
        print(json.dumps(Config._config, indent=4, sort_keys=True))
