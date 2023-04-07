
class LabelItem:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def getName(self) -> str:
        return self.name

    def getDescription(self) -> str:
        return self.description
