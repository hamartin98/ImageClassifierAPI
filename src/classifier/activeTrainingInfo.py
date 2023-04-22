from enum import IntEnum
import datetime
import json
from typing import List, Dict, Any

from .config.classifierConfig import ClassifierConfig
from .config.config import Config
from .utils.timeUtils import TimeUtils
from .utils.imagePlotterUtils import ImagePlotterUtils


class TrainingStatus(IntEnum):
    '''Training status enum'''

    NONE = 0
    STARTED = 1
    RUNNING = 2
    FINISHED = 3
    TESTING = 4
    ERROR = 5
    INTERRUPTED = 6
    CANCELLED = 7

    def __str__(self) -> str:
        '''Get current value as a string'''

        return str(self.name).lower()


class ActiveTrainingInfo:
    '''Follow the status of the currently running training'''

    name: str = ''
    savePath: str = ''
    saveResult: bool = False
    status: TrainingStatus = TrainingStatus.NONE
    startTime = None
    endTime = None
    totalEpochs: int = None
    currentEpoch: int = None
    isLogging: bool = False
    errorMessage: str = None
    config: ClassifierConfig = None
    runninglossData: List[float] = []
    runningAccuracyData: List[float] = []
    epochTimes: List[str] = []
    accuracyPerClass: Dict[str, float] = {}

    def __init__(self) -> None:
        pass

    @staticmethod
    def start() -> None:
        '''Start the current trainig session'''

        activeStatus = ActiveTrainingInfo.getStatus()
        if activeStatus in [TrainingStatus.RUNNING, TrainingStatus.STARTED]:
            ActiveTrainingInfo.setStatus(TrainingStatus.INTERRUPTED)

        ActiveTrainingInfo.reset()
        ActiveTrainingInfo.status = TrainingStatus.STARTED

        ActiveTrainingInfo.status = TrainingStatus.RUNNING

    @staticmethod
    def stop() -> None:
        '''Stop the currently running training session'''

        ActiveTrainingInfo.status = TrainingStatus.FINISHED

    @staticmethod
    def reset() -> None:
        '''Reset active status information'''

        ActiveTrainingInfo.name = ''
        ActiveTrainingInfo.savePath = ''
        ActiveTrainingInfo.saveResult = False
        ActiveTrainingInfo.startTime = None
        ActiveTrainingInfo.endTime = None
        ActiveTrainingInfo.totalEpochs = None
        ActiveTrainingInfo.currentEpoch = None
        ActiveTrainingInfo.isLogging = False
        ActiveTrainingInfo.config = None
        ActiveTrainingInfo.runninglossData = []
        ActiveTrainingInfo.runningAccuracyData = []
        ActiveTrainingInfo.epochTimes = []
        ActiveTrainingInfo.accuracyPerClass = {}

    @staticmethod
    def setStartTime(startTime) -> None:
        '''Set start time of the current training'''

        ActiveTrainingInfo.startTime = startTime

    @staticmethod
    def getStartTime() -> None:
        '''Get the start time of the current training'''

        return ActiveTrainingInfo.startTime

    @staticmethod
    def getStartTimeStr() -> str:
        '''Get start time of the current training as a string'''

        if ActiveTrainingInfo.getStartTime():
            return TimeUtils.toString(ActiveTrainingInfo.getStartTime())

        return ''

    @staticmethod
    def setEndTime(endTime) -> None:
        '''Set end time of the current training'''

        ActiveTrainingInfo.endTime = endTime

    @staticmethod
    def getEndTime() -> None:
        '''Get end time of the current training'''

        return ActiveTrainingInfo.endTime

    @staticmethod
    def getEndTimeStr() -> str:
        '''Get end time of the current training as a string'''

        if ActiveTrainingInfo.getEndTime():
            return TimeUtils.toString(ActiveTrainingInfo.getEndTime())

        return ''

    @staticmethod
    def getCurrentTime() -> datetime:
        '''Get the current runtime of the active training'''

        if ActiveTrainingInfo.getStartTime() != None:
            currentTime = TimeUtils.getCurrentTime()
            return currentTime - ActiveTrainingInfo.getStartTime()

        return None

    @staticmethod
    def getCurrentTimeStr() -> str:
        '''Get the current runtime of the active training as a string'''

        if ActiveTrainingInfo.getCurrentTime():
            return TimeUtils.toString(ActiveTrainingInfo.getCurrentTime())

        return ''

    @staticmethod
    def getTotalTime() -> None:
        '''Get the total runtime of the current training'''

        if ActiveTrainingInfo.getEndTime() != None and ActiveTrainingInfo.getStartTime() != None:
            return ActiveTrainingInfo.endTime - ActiveTrainingInfo.startTime

        return ActiveTrainingInfo.getCurrentTime()

    @staticmethod
    def getTotalTimeStr() -> str:
        '''Get the total runtime of the current training as a string'''

        return TimeUtils.toString(ActiveTrainingInfo.getTotalTime())

    @staticmethod
    def setStatus(status: TrainingStatus) -> None:
        '''Set active trainig status'''

        ActiveTrainingInfo.status = status

    @staticmethod
    def getStatus() -> TrainingStatus:
        '''Get active training status'''

        return ActiveTrainingInfo.status

    @staticmethod
    def getStatusStr() -> str:
        '''Get active training status as a string'''

        return str(ActiveTrainingInfo.status)

    @staticmethod
    def setTotalEpochs(epochs: int) -> None:
        '''Set the total number of epochs'''

        ActiveTrainingInfo.totalEpochs = epochs

    @staticmethod
    def getTotalEpochs() -> int:
        '''Get the total number of epochs'''

        return ActiveTrainingInfo.totalEpochs

    @staticmethod
    def setCurrentEpochs(epochs) -> None:
        '''Set the number of current epochs'''

        ActiveTrainingInfo.currentEpoch = epochs

    @staticmethod
    def stepCurrentEpochs(step: int = 1) -> None:
        '''Increase the number of current epochs with the given step value'''

        ActiveTrainingInfo.currentEpoch += step

    @staticmethod
    def getCurrentEpochs() -> int:
        '''Get the number of current epochs'''

        return ActiveTrainingInfo.currentEpoch

    @staticmethod
    def setLoggingStatus(logging: bool) -> None:
        '''Set the logging status'''

        ActiveTrainingInfo.isLogging = logging

    @staticmethod
    def getLoggingStatus() -> bool:
        '''Get the logging status'''

        return ActiveTrainingInfo.isLogging

    @staticmethod
    def setErrorMessage(message: str) -> None:
        '''Set error message'''

        ActiveTrainingInfo.errorMessage = message

    @staticmethod
    def getErrorMessage() -> str:
        '''Get error message'''

        return ActiveTrainingInfo.errorMessage

    @staticmethod
    def setError(message: str) -> None:
        '''Set error status with the given error message'''

        ActiveTrainingInfo.status = TrainingStatus.ERROR
        ActiveTrainingInfo.setErrorMessage(message)

    @staticmethod
    def setConfig(config: ClassifierConfig) -> None:
        '''Set active training session's configuration data'''

        ActiveTrainingInfo.config = config

    @staticmethod
    def getConfig() -> ClassifierConfig:
        '''Get active training's configuration data'''

        return ActiveTrainingInfo.config

    @staticmethod
    def getRunningLoss() -> List[float]:
        '''Get the list of collected running loss values'''

        return ActiveTrainingInfo.runninglossData

    @staticmethod
    def setRunningLoss(runningLossData: List[float]) -> None:
        '''Set the list of collected running loss values'''

        ActiveTrainingInfo.runninglossData = runningLossData

    @staticmethod
    def addRunningLossData(lossData: float) -> None:
        '''Add new value to the list of running loss values'''

        ActiveTrainingInfo.runninglossData.append(lossData)

    @staticmethod
    def getRunningAccuracy() -> List[float]:
        '''Get the list of collected running accuracy values'''

        return ActiveTrainingInfo.runningAccuracyData

    @staticmethod
    def setRunningAccuracy(runningAccuracyData: List[float]) -> None:
        '''Set the list of collected running accuracy values'''

        ActiveTrainingInfo.runningAccuracyData = runningAccuracyData

    @staticmethod
    def addRunningAccuracyData(accuracyData: float) -> None:
        '''Add new value to the list of running accuracy values'''

        ActiveTrainingInfo.runningAccuracyData.append(accuracyData)

    @staticmethod
    def getAccuracyPerClass() -> Dict[str, float]:
        '''Get accuracy values for each class'''

        return ActiveTrainingInfo.accuracyPerClass

    @staticmethod
    def setAccuracyPerClass(accuracyPerClass: Dict[str, float]) -> None:
        '''Set accuracy values for each class'''

        ActiveTrainingInfo.accuracyPerClass = accuracyPerClass

    @staticmethod
    def addRunningLossAndAccuracy(lossData: float, accuracyData: float) -> None:
        '''Add loss and accuracy data to the corresponding lists'''

        ActiveTrainingInfo.addRunningLossData(lossData)
        ActiveTrainingInfo.addRunningAccuracyData(accuracyData)

    @staticmethod
    def getEpochTimes() -> List[str]:
        '''Get the list of epoch durations'''

        return ActiveTrainingInfo.epochTimes

    @staticmethod
    def setEpochTimes(epochTimes: List[str]) -> None:
        '''Set the list of epoch durations'''

        ActiveTrainingInfo.epochTimes = epochTimes

    @staticmethod
    def addEpochTime(epochTime: str) -> None:
        '''Add a new value to the list of epoch durations'''

        ActiveTrainingInfo.epochTimes.append(epochTime)

    @staticmethod
    def getName() -> str:
        '''Get the name of the current trainig session'''

        return ActiveTrainingInfo.name

    @staticmethod
    def setName(name: str) -> None:
        '''Set the name of the current training session'''

        ActiveTrainingInfo.name = name

    @staticmethod
    def getSavePath() -> str:
        '''Get path to save the current training information'''

        return ActiveTrainingInfo.savePath

    @staticmethod
    def setSavePath(savePath: str) -> None:
        '''Set the path to save the current trainig information to'''

        ActiveTrainingInfo.savePath = savePath

    @staticmethod
    def canStartNew() -> bool:
        '''Return whether a new training session can be started'''

        status = ActiveTrainingInfo.getStatus()
        isActive = (status == TrainingStatus.STARTED or status ==
                    TrainingStatus.RUNNING or status == TrainingStatus.TESTING)

        return not isActive

    @staticmethod
    def saveLossAndAccuracyDiagram(basePath: str, name: str) -> None:
        '''Create and save loss and accuracy diagram from the collected data'''

        savePath = f'{basePath}/{name}.png'
        lossData = ActiveTrainingInfo.getRunningLoss()
        accuracyData = ActiveTrainingInfo.getRunningAccuracy()
        ImagePlotterUtils.plotLossAndAccuracy(lossData, accuracyData, savePath)

    @staticmethod
    def saveResultJson(basePath: str, name: str) -> None:
        '''Save training summary as a json file'''

        savePath = f'{basePath}/{name}.json'

        with open(savePath, 'w') as outFile:
            jsonData = ActiveTrainingInfo.toJson()
            json.dump(jsonData, outFile, indent=4, sort_keys=False)

    @staticmethod
    def saveResultData() -> None:
        '''Save training result'''

        basePath = Config.getLearningInfoPath()
        name = ActiveTrainingInfo.getName()
        ActiveTrainingInfo.saveLossAndAccuracyDiagram(basePath, name)
        ActiveTrainingInfo.saveResultJson(basePath, name)

    @staticmethod
    def createEpochInfo() -> None:
        '''Summarize epoch informations'''

        result: List[Dict[str, Any]] = []

        epochTimes = ActiveTrainingInfo.getEpochTimes()
        loss = ActiveTrainingInfo.getRunningLoss()
        accuracy = ActiveTrainingInfo.getRunningAccuracy()

        for epoch in range(len(loss)):
            epochInfo = {
                'epoch': epoch + 1,
                'time': epochTimes[epoch],
                'loss': loss[epoch],
                'accuracy': accuracy[epoch] * 100
            }

            result.append(epochInfo)

        return result

    @staticmethod
    def toJson() -> Dict:
        '''Convert training summary to json'''

        config: ClassifierConfig = ActiveTrainingInfo.getConfig()
        configJson = {}
        if config:
            configJson = config.getAsJson()

        result = {
            'name': ActiveTrainingInfo.getName(),
            'meta': {
                'status': ActiveTrainingInfo.getStatusStr(),
                'errorMessage': ActiveTrainingInfo.getErrorMessage(),
                'startTime': ActiveTrainingInfo.getStartTimeStr(),
                'endTime': ActiveTrainingInfo.getEndTimeStr(),
                'runTime': ActiveTrainingInfo.getCurrentTimeStr(),
                'config': configJson
            },
            'progress': {
                'currentEpoch': ActiveTrainingInfo.getCurrentEpochs(),
                'totalEpoch': ActiveTrainingInfo.getTotalEpochs(),
                'epochData': ActiveTrainingInfo.createEpochInfo()
            },
            'accuracy': ActiveTrainingInfo.getAccuracyPerClass()
        }

        return result

    @staticmethod
    def print() -> None:
        '''Print the training information as a formatted json'''

        data = ActiveTrainingInfo.toJson()
        formattedData = json.dumps(data, indent=4, sort_keys=False)
        print(formattedData)
