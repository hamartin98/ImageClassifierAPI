from enum import IntEnum


class TrainingStatus(IntEnum):
    NONE = 0
    STARTED = 1
    RUNNING = 2
    FINISHED = 3
    ERROR = 4
    INTERRUPTED = 5
    CANCELLED = 6


class ActiveTrainingInfo:
    status: TrainingStatus = TrainingStatus.NONE
    startTime = None
    endTime = None
    totalEpochs: int = None
    currentEpoch: int = None
    isLogging: bool = False
    errorMessage: str = None
    # TODO: Loss and accuracy information

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
        ActiveTrainingInfo.startTime = None
        ActiveTrainingInfo.endTime = None
        ActiveTrainingInfo.totalEpochs = None
        ActiveTrainingInfo.currentEpoch = None
        ActiveTrainingInfo.isLogging = False

    @staticmethod
    def setStartTime(startTime) -> None:
        ActiveTrainingInfo.startTime = startTime

    @staticmethod
    def getStartTime() -> None:
        return ActiveTrainingInfo.startTime

    @staticmethod
    def setEndTime(endTime) -> None:
        ActiveTrainingInfo.endTime = endTime

    @staticmethod
    def getEndTime() -> None:
        return ActiveTrainingInfo.endTime

    @staticmethod
    def getCurrentTime() -> None:
        # TODO: Implement
        pass

    @staticmethod
    def getTotalTime() -> None:
        return ActiveTrainingInfo.endTime - ActiveTrainingInfo.startTime

    @staticmethod
    def setStatus(status: TrainingStatus) -> None:
        ActiveTrainingInfo.status = status

    @staticmethod
    def getStatus() -> TrainingStatus:
        return ActiveTrainingInfo.status

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
