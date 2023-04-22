import json
import os
from typing import Any


class Config:
    '''Base configuration class'''

    _inited = False
    _path = None
    _config = {
        'basePath': 'data',
        'isRelativePath': True
    }

    def __init__(self, fileName) -> None:
        '''Basic initialization'''

        Config._path = fileName
        if not Config._inited:
            Config.setFromFile(fileName)
            Config.overrideFromEnv()
            Config.print()
            Config._inited = True

    @staticmethod
    def isSet(value: Any) -> bool:
        '''Return whether the given value is set'''

        return value is not None and value != ''

    @staticmethod
    def setFromFile(fileName: str) -> None:
        '''Set configuration from the given file'''

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
        '''Override values from environment'''

        Config.setBasePath(os.environ.get('BASE_PATH'))
        Config.setIsRelativePath(os.environ.get('IS_RELATIVE_PATH'))

    @staticmethod
    def getBasePath() -> None:
        '''Return base path to load from and save to data'''

        return Config._config['basePath']

    @staticmethod
    def setBasePath(newValue: str) -> None:
        '''Set base path value'''

        if Config.isSet(newValue):
            Config._config['basePath'] = newValue

    @staticmethod
    def getIsRelativePath() -> None:
        '''Return whether the given path is relative or absolute'''

        return Config._config['isRelativePath']

    @staticmethod
    def setIsRelativePath(newValue: bool) -> None:
        '''Set whether the given path is relative or absolute'''

        if Config.isSet(newValue):
            Config._config['isRelativePath'] = newValue

    @staticmethod
    def getImagesPath() -> None:
        '''Return base path to the images data'''

        path = os.path.join(Config._config['basePath'], 'images')
        if Config.getIsRelativePath():
            return os.path.realpath(path)
        return os.path.abspath(path)

    @staticmethod
    def getModelsPath() -> None:
        '''Return base path to the models data'''

        path = os.path.join(Config._config['basePath'], 'models')
        if Config.getIsRelativePath():
            return os.path.realpath(path)
        return os.path.abspath(path)

    @staticmethod
    def getLearningInfoPath() -> None:
        '''Return base path to save learning information'''

        path = os.path.join(Config._config['basePath'], 'learningInfo')
        if Config.getIsRelativePath():
            return os.path.realpath(path)
        return os.path.abspath(path)

    @staticmethod
    def getPath() -> None:
        '''Get path'''

        return Config._path

    @staticmethod
    def print() -> None:
        '''Print the configuration as a formatted json'''

        print(json.dumps(Config._config, indent=4, sort_keys=True))
