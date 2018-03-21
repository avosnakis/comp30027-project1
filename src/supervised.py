"""
Module for all functions driving the supervised classifier.
"""

from typing import List, Tuple

from ClassifierData import ClassifierData
from SupervisedNBClassifier import SupervisedNBClassifier


def train_supervised(data: ClassifierData) -> SupervisedNBClassifier:
    """
    Instantiates a new supervised Naive Bayes Classifier and trains it.
    :param data: The data to be fed into the classifier.
    :return: A trained supervised NB classifier.
    """
    classifier: SupervisedNBClassifier = SupervisedNBClassifier()
    classifier.train(data)
    return classifier


def predict_supervised(classifier: SupervisedNBClassifier, data: List[List[str]]) -> None:
    """
    Predicts the classes for a set of instances.
    :param classifier: The trained classifier.
    :param data: The instances to classify.
    """
    predicted: List[Tuple[str, List[str]]] = classifier.predict_set(data)
    for pred in predicted:
        print("Predicted class: {}, instance: {}".format(pred[0], pred[1]))


def evaluate_supervised(classifier: SupervisedNBClassifier, data: List[List[str]]) -> None:
    """
    Evaluates a classifier against a set of data, printing out the percentage of correct
    classifications.
    :param classifier: The trained classifier.
    :param data: The data holing the testing data.
    """
    percentage_correct: float = classifier.evaluate(data)
    print("Percentage of correct classifications: {:.6}%".format(percentage_correct))
