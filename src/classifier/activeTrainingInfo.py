from enum import IntEnum
from typing import List, Dict, Any
import json
import threading

from .config.classifierConfig import ClassifierConfig
from .utils.timeUtils import TimeUtils
from .utils.imagePlotterUtils import ImagePlotterUtils
from .config.config import Config


class TrainingStatus(IntEnum):
    NONE = 0
    STARTED = 1
    RUNNING = 2
    FINISHED = 3
    TESTING = 4
    ERROR = 5
    INTERRUPTED = 6
    CANCELLED = 7

    def __str__(self) -> str:
        return str(self.name).lower()


class ActiveTrainingInfo:
    lock = threading.Lock()
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

    # TODO: Log progress

    def __init__(self) -> None:
        pass

    @staticmethod
    def start() -> None:
        activeStatus = ActiveTrainingInfo.getStatus()
        if activeStatus in [TrainingStatus.RUNNING, TrainingStatus.STARTED]:
            ActiveTrainingInfo.setStatus(TrainingStatus.INTERRUPTED)

        ActiveTrainingInfo.reset()
        ActiveTrainingInfo.status = TrainingStatus.STARTED

        ActiveTrainingInfo.status = TrainingStatus.RUNNING

    @staticmethod
    def stop() -> None:
        ActiveTrainingInfo.status = TrainingStatus.FINISHED

    @staticmethod
    def reset() -> None:
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
        ActiveTrainingInfo.startTime = startTime

    @staticmethod
    def getStartTime() -> None:
        return ActiveTrainingInfo.startTime

    @staticmethod
    def getStartTimeStr() -> str:
        if ActiveTrainingInfo.getStartTime():
            return TimeUtils.toString(ActiveTrainingInfo.getStartTime())

        return ''

    @staticmethod
    def setEndTime(endTime) -> None:
        ActiveTrainingInfo.endTime = endTime

    @staticmethod
    def getEndTime() -> None:
        return ActiveTrainingInfo.endTime

    @staticmethod
    def getEndTimeStr() -> str:
        if ActiveTrainingInfo.getEndTime():
            return TimeUtils.toString(ActiveTrainingInfo.getEndTime())

        return ''

    @staticmethod
    def getCurrentTime() -> None:
        if ActiveTrainingInfo.getStartTime() != None:
            currentTime = TimeUtils.getCurrentTime()
            return currentTime - ActiveTrainingInfo.getStartTime()

        return None

    @staticmethod
    def getCurrentTimeStr() -> str:
        if ActiveTrainingInfo.getCurrentTime():
            return TimeUtils.toString(ActiveTrainingInfo.getCurrentTime())

        return ''

    @staticmethod
    def getTotalTime() -> None:
        if ActiveTrainingInfo.getEndTime() != None and ActiveTrainingInfo.getStartTime() != None:
            return ActiveTrainingInfo.endTime - ActiveTrainingInfo.startTime

        return ActiveTrainingInfo.getCurrentTime()

    @staticmethod
    def getTotalTimeStr() -> str:
        return TimeUtils.toString(ActiveTrainingInfo.getTotalTime())

    @staticmethod
    def setStatus(status: TrainingStatus) -> None:
        ActiveTrainingInfo.status = status

    @staticmethod
    def getStatus() -> TrainingStatus:
        return ActiveTrainingInfo.status

    @staticmethod
    def getStatusStr() -> str:
        return str(ActiveTrainingInfo.status)

    @staticmethod
    def setTotalEpochs(epochs: int) -> None:
        ActiveTrainingInfo.totalEpochs = epochs

    @staticmethod
    def getTotalEpochs() -> int:
        return ActiveTrainingInfo.totalEpochs

    @staticmethod
    def setCurrentEpochs(epochs) -> None:
        ActiveTrainingInfo.currentEpoch = epochs

    @staticmethod
    def stepCurrentEpochs(step: int = 1) -> None:
        ActiveTrainingInfo.currentEpoch += step

    @staticmethod
    def getCurrentEpochs() -> int:
        return ActiveTrainingInfo.currentEpoch

    @staticmethod
    def setLoggingStatus(logging: bool) -> None:
        ActiveTrainingInfo.isLogging = logging

    @staticmethod
    def getLoggingStatus() -> bool:
        return ActiveTrainingInfo.isLogging

    @staticmethod
    def setErrorMessage(message: str) -> None:
        ActiveTrainingInfo.errorMessage = message

    @staticmethod
    def getErrorMessage() -> str:
        return ActiveTrainingInfo.errorMessage

    @staticmethod
    def setError(message: str) -> None:
        ActiveTrainingInfo.status = TrainingStatus.ERROR
        ActiveTrainingInfo.setErrorMessage(message)

    @staticmethod
    def setConfig(config: ClassifierConfig) -> None:
        ActiveTrainingInfo.config = config

    @staticmethod
    def getConfig() -> None:
        return ActiveTrainingInfo.config

    @staticmethod
    def getRunningLoss() -> List[float]:
        return ActiveTrainingInfo.runninglossData

    @staticmethod
    def setRunningLoss(runningLossData: List[float]) -> None:
        ActiveTrainingInfo.runninglossData = runningLossData

    @staticmethod
    def addRunningLossData(lossData: float) -> None:
        ActiveTrainingInfo.runninglossData.append(lossData)

    @staticmethod
    def getRunningAccuracy() -> List[float]:
        return ActiveTrainingInfo.runningAccuracyData

    @staticmethod
    def setRunningAccuracy(runningAccuracyData: List[float]) -> None:
        ActiveTrainingInfo.runningAccuracyData = runningAccuracyData

    @staticmethod
    def addRunningAccuracyData(accuracyData: float) -> None:
        ActiveTrainingInfo.runningAccuracyData.append(accuracyData)

    @staticmethod
    def getAccuracyPerClass() -> Dict[str, float]:
        return ActiveTrainingInfo.accuracyPerClass

    @staticmethod
    def setAccuracyPerClass(accuracyPerClass: Dict[str, float]) -> None:
        ActiveTrainingInfo.accuracyPerClass = accuracyPerClass

    @staticmethod
    def addRunningLossAndAccuracy(lossData: float, accuracyData: float) -> None:
        ActiveTrainingInfo.addRunningLossData(lossData)
        ActiveTrainingInfo.addRunningAccuracyData(accuracyData)

    @staticmethod
    def getEpochTimes() -> List[str]:
        return ActiveTrainingInfo.epochTimes

    @staticmethod
    def setEpochTimes(epochTimes: List[str]) -> None:
        ActiveTrainingInfo.epochTimes = epochTimes

    @staticmethod
    def addEpochTime(epochTime: str) -> None:
        ActiveTrainingInfo.epochTimes.append(epochTime)

    @staticmethod
    def getName() -> str:
        return ActiveTrainingInfo.name

    @staticmethod
    def setName(name: str) -> None:
        ActiveTrainingInfo.name = name

    @staticmethod
    def getSavePath() -> str:
        return ActiveTrainingInfo.savePath

    @staticmethod
    def setSavePath(savePath: str) -> None:
        ActiveTrainingInfo.savePath = savePath

    @staticmethod
    def canStartNew() -> bool:
        status = ActiveTrainingInfo.getStatus()
        isActive = (status == TrainingStatus.STARTED or status ==
                    TrainingStatus.RUNNING or status == TrainingStatus.TESTING)

        return not isActive

    @staticmethod
    def saveLossAndAccuracyDiagram(basePath: str, name: str) -> None:
        savePath = f'{basePath}/{name}.png'
        lossData = ActiveTrainingInfo.getRunningLoss()
        accuracyData = ActiveTrainingInfo.getRunningAccuracy()
        ImagePlotterUtils.plotLossAndAccuracy(lossData, accuracyData, savePath)

    @staticmethod
    def saveResultJson(basePath: str, name: str) -> None:
        savePath = f'{basePath}/{name}.json'

        with open(savePath, 'w') as outFile:
            jsonData = ActiveTrainingInfo.toJson()
            json.dump(jsonData, outFile, indent=4, sort_keys=False)

    @staticmethod
    def saveResultData() -> None:
        basePath = Config.getLearningInfoPath()
        name = ActiveTrainingInfo.getName()
        ActiveTrainingInfo.saveLossAndAccuracyDiagram(basePath, name)
        ActiveTrainingInfo.saveResultJson(basePath, name)

    @staticmethod
    def createEpochInfo() -> None:
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
        data = ActiveTrainingInfo.toJson()
        formattedData = json.dumps(data, indent=4, sort_keys=False)
        print(formattedData)
