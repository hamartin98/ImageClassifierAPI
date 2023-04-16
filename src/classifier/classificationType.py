from enum import IntEnum


class ClassificationType(IntEnum):
    NONE = 0
    BUILDING = 1
    VEGETATION = 2
    ROAD = 3

    def __str__(self) -> str:
        return str(self.name).lower()


class ClassificationTypeUtils:
    @staticmethod
    def fromString(value: str) -> ClassificationType:
        normValue = value.lower()
        if normValue == 'building':
            return ClassificationType.BUILDING
        elif normValue == 'vegetation':
            return ClassificationType.VEGETATION
        elif normValue == 'road':
            return ClassificationType.ROAD
        else:
            return ClassificationType.NONE
