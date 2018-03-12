from ClassifierData import ClassifierData
from SupervisedNBClassifier import SupervisedNBClassifier


def evaluate_supervised(classifier: SupervisedNBClassifier, data: ClassifierData) -> None:
    percentage_correct: float = classifier.evaluate(data.get_training_data())
    print("Percentage of correct classifications: {:.6}%".format(percentage_correct))
