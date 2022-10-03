from typing import Dict
from river.datasets.base import SyntheticDataset
import random


class BinaryImbalancedDataset:
    def __init__(self, synth_generator: SyntheticDataset, class_balance: float):
        self.generator = iter(synth_generator)
        self.randomClassifier = random.Random(42)
        self.class_balance = class_balance
        self.idx = 0
        pass

    def take(self, k: int):
        while True:
            if self.idx > k:
                return "error"
            expected_class = 1
            if self.randomClassifier.uniform(0, 1) > self.class_balance:
                expected_class = 0
            x, y = next(self.generator)
            while y != expected_class:
                x, y = next(self.generator)

            self.idx += 1

            yield x, y


class MultiClassImbalancedDataset:
    def __init__(self, synth_generator: SyntheticDataset, class_balance: list):
        self.generator = iter(synth_generator)
        self.randomClassifier = random.Random(42)
        self.class_balance = class_balance
        self.idx = 0
        pass

    def take(self, k: int):
        while True:
            if self.idx > k:
                return "error"
            expected_class = -1
            p = self.randomClassifier.uniform(0, 1)
            while p > 0:
                expected_class += 1
                p -= self.class_balance[expected_class]

            x, y = next(self.generator)
            while y != expected_class:
                x, y = next(self.generator)

            self.idx += 1

            yield x, y
