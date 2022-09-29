from evaluator import Evaluator
from river.base import Classifier
from river.datasets.base import Dataset
import pandas as pd
from tqdm import tqdm


class Experiment:
    def __init__(self, output_folder: str, number_of_processes: int = 1) -> None:

        self.output_folder = output_folder
        self.number_of_processes = number_of_processes
        self.experiments = []

    def add(
        self, model: Classifier, data: Dataset, exp_name: str, evaluator: Evaluator
    ) -> None:
        self.experiments.append((model, data, exp_name, evaluator))

    def run(self) -> None:
        for exp in self.experiments:
            results = []
            model = exp[0]
            dataset = exp[1]
            exp_name = exp[2]
            evaluator = exp[3]
            idx = 0
            for x, y in dataset:
                model.learn_one(x, y)
                predictions = model.predict_one(x)
                evaluator.add(y, predictions)
                idx += 1
                if idx % evaluator.getWindowSize() == 0:
                    metrics = evaluator.getMetrics()
                    metrics["classified instances"] = idx
                    results.append(metrics)

            result_df = pd.DataFrame(results)
            file_name = "{}/{}.csv".format(self.output_folder, exp_name)
            result_df.to_csv(file_name, index=None)
