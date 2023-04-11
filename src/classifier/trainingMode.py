from enum import IntEnum


# TODO: Use this to determine traing mode
class TrainingMode(IntEnum):
    NONE = 0
    TRAIN = 1
    TEST = 2
    CLASSIFY = 3
    TRAIN_AND_TEST = 4
