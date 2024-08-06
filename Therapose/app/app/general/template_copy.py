import copy

class TemplateCopy:
    def __init__(self):
        pass
    
    def copy(self):
        return copy.deepcopy(self)