from evaluator import Evaluator
from experiment import Experiment
from river.tree import HoeffdingAdaptiveTreeClassifier, HoeffdingTreeClassifier
from river.ensemble import AdaptiveRandomForestClassifier
from river.datasets import synth
from generators.imbGenerator import BinaryImbalancedDataset, MultiClassImbalancedDataset

# ideal would be something like
# evaluator = Evaluator (window_size = 500, metrics = X)
# exp = Experiment(evaluator, number_of_parallel = 8, output_folder = X)
# exp.add(classifier, stream, name = "X")
# exp.add(classifier, stream, name = "Y")
# exp.add(classifier, stream, name = "Z")


def main():

    exp = Experiment(number_of_processes=1, output_folder=".")
    exp.add(
        HoeffdingTreeClassifier(),
        synth.LED().take(10000),
        exp_name="HT-aggraval",
        evaluator=Evaluator(),
    )
    exp.add(
        HoeffdingAdaptiveTreeClassifier(),
        synth.LED().take(10000),
        exp_name="HAT-aggraval",
        evaluator=Evaluator(),
    )
    exp.add(
        AdaptiveRandomForestClassifier(),
        synth.LED().take(10000),
        exp_name="ARF-aggraval",
        evaluator=Evaluator(),
    )
    exp.run()


if __name__ == "__main__":
    main()
