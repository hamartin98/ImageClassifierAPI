from enum import IntEnum


class ClassificationType(IntEnum):
    '''Classification types'''

    NONE = 0
    BUILDING = 1
    VEGETATION = 2
    ROAD = 3

    def __str__(self) -> str:
        return str(self.name).lower()


class ClassificationTypeUtils:
    '''Collection of classification type related utility functions'''

    @staticmethod
    def fromString(value: str) -> ClassificationType:
        '''Convert string to classification type'''

        normValue = value.lower()
        if normValue == 'building':
            return ClassificationType.BUILDING
        elif normValue == 'vegetation':
            return ClassificationType.VEGETATION
        elif normValue == 'road':
            return ClassificationType.ROAD
        else:
            return ClassificationType.NONE
