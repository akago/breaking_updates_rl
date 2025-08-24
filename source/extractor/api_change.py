
class ApiChange:

    def __init__(self, element: str, kind: str, newVersion: str, oldVersion: str):
        self.element = element
        self.kind = kind
        
        self.newVersion = newVersion
        self.oldVersion = oldVersion


