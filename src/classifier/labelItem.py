
class LabelItem:
    '''Label information class'''

    def __init__(self, name: str, description: str):
        '''Init label item'''

        self.name = name
        self.description = description

    def getName(self) -> str:
        '''Get label's name'''

        return self.name

    def getDescription(self) -> str:
        '''Get label's description'''

        return self.description
