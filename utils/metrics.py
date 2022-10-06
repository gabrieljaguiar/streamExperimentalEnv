import time

class TimeConsumption():
    def __init__(self) -> None:
        self.startingTime = time.time()
    
    def update(self, trueClass, predictedClass):
        self.actualTime = time.time()
    
    def get(self):
        return self.actualTime - self.startingTime
    
    