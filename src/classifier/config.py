import json
import os
from typing import Any

class Config:
    _inited = False
    _path = None
    _config = {
            'basePath': 'data',
            'isRelativePath': True
        }

    def __init__(self, fileName) -> None:
        
        Config._path = fileName
        if not Config._inited:
            Config.setFromFile(fileName)
            Config.overrideFromEnv()
            Config.print()
            Config._inited = True
            
    @staticmethod
    def isSet(value: Any) -> bool:
        return value is not None and value != ''

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
        if Config.isSet(newValue):
            Config._config['basePath'] = newValue

    @staticmethod
    def getIsRelativePath() -> None:
        return Config._config['isRelativePath']

    @staticmethod
    def setIsRelativePath(newValue) -> None:
        if Config.isSet(newValue):
            Config._config['isRelativePath'] = newValue

    @staticmethod
    def getImagesPath() -> None:
        path = os.path.join(Config._config['basePath'], 'images')
        if Config.getIsRelativePath():
            return os.path.realpath(path)
        return os.path.abspath(path)

    @staticmethod
    def getModelsPath() -> None:
        path = os.path.join(Config._config['basePath'], 'models')
        if Config.getIsRelativePath():
            return os.path.realpath(path)
        return os.path.abspath(path)
    
    @staticmethod
    def getPath() -> None:
        return Config._path

    @staticmethod
    def print() -> None:
        print(json.dumps(Config._config, indent=4, sort_keys=True))
