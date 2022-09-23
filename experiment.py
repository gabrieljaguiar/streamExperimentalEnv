from evaluator import Evaluator
from river.base import Classifier
from river.datasets.base import Dataset

class Experiment():
    def __init__(self, eval: Evaluator, output_folder:str, number_of_processes: int = 1) -> None:
        self.evaluator = eval
        self.output_folder = output_folder
        self.number_of_processes = number_of_processes
        self.experiments = []


    def add(self, model: Classifier, data: Dataset, exp_name: str) -> None:
        self.experiments.append((model, data, exp_name))
    

    def run() ->  None:
        pass