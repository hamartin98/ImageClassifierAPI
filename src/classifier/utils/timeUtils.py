from datetime import datetime, timedelta


class TimeUtils:
    @staticmethod
    def getCurrentTime() -> datetime:
        dateTime = datetime.now()

        return dateTime

    @staticmethod
    def getCurrentTimeStamp() -> float:
        dateTime = datetime.now()
        timeStamp = datetime.timestamp(dateTime)

        return timeStamp

    @staticmethod
    def getCurrentTimeStr() -> str:
        timeStamp = TimeUtils.getCurrentTimeStamp()
        return str(timedelta(seconds=round(timeStamp)))

    @staticmethod
    def getTimeDiffStr(startTime: float, endTime: float) -> str:
        timeDiff = endTime - startTime
        timeStr = str(timedelta(seconds=round(timeDiff)))

        return timeStr

    @staticmethod
    def toString(dateTime) -> str:
        if isinstance(dateTime, timedelta):
            rounded = round(dateTime.total_seconds())
            return str(timedelta(seconds=rounded))

        return dateTime.strftime("%Y-%m-%dT%H:%M:%S")
