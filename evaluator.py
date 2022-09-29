from typing import Dict
from river import metrics


class Evaluator:
    def __init__(self, window_size=500, metric_list=None) -> None:
        self.windowMetrics = []
        self.windowSize = window_size

        self.metrics = {
            "accuracy": metrics.Accuracy,
            "kappa": metrics.CohenKappa,
            "gmean": metrics.GeometricMean,
            "cm": metrics.ConfusionMatrix,
        }

        if metric_list is None:
            metric_list = self.metrics.keys()

        self.metric_list = metric_list

        for m in metric_list:
            metric_function = self.metrics.get(m)
            self.windowMetrics.append([m, metric_function()])

    def add(self, trueClass: int, predictedClass: int) -> None:
        for m in self.windowMetrics:
            m[1].update(trueClass, predictedClass)

    def getMetrics(self) -> Dict:
        computedMetrics = {}
        for m in self.windowMetrics:
            if m[0] != "cm":
                computedMetrics[m[0]] = m[1].get()
            else:
                classes = m[1].classes
                for i in range(0, len(classes)):
                    for j in range(0, len(classes)):
                        computedMetrics["cm[{}][{}]".format(i, j)] = m[1].data[i][j]

        self.reset()
        return computedMetrics

    def getWindowSize(self) -> int:
        return self.windowSize

    def reset(self):
        self.windowMetrics = []
        for m in self.metric_list:
            metric_function = self.metrics.get(m)
            self.windowMetrics.append([m, metric_function()])
