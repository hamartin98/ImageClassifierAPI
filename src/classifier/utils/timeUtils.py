from datetime import datetime, timedelta


class TimeUtils:
    '''Collection of time handling utility functions'''

    @staticmethod
    def getCurrentTime() -> datetime:
        '''Return current datetime'''

        dateTime = datetime.now()

        return dateTime

    @staticmethod
    def getCurrentTimeStamp() -> float:
        '''Return current timestamp'''

        dateTime = datetime.now()
        timeStamp = datetime.timestamp(dateTime)

        return timeStamp

    @staticmethod
    def getCurrentTimeStr() -> str:
        '''Return current time as a string'''

        timeStamp = TimeUtils.getCurrentTimeStamp()
        return str(timedelta(seconds=round(timeStamp)))

    @staticmethod
    def getTimeDiffStr(startTime: float, endTime: float) -> str:
        '''Return time difference as a string'''

        timeDiff = endTime - startTime
        timeStr = str(timedelta(seconds=round(timeDiff)))

        return timeStr

    @staticmethod
    def toString(dateTime) -> str:
        '''Convert the given time to string'''

        if isinstance(dateTime, timedelta):
            rounded = round(dateTime.total_seconds())
            return str(timedelta(seconds=rounded))

        return dateTime.strftime("%Y-%m-%dT%H:%M:%S")
