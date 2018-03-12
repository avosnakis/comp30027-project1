from typing import List, Tuple

from SupervisedNBClassifier import SupervisedNBClassifier
from ClassifierData import ClassifierData


def predict_supervised(classifier: SupervisedNBClassifier, data: ClassifierData) -> None:
    predicted: List[Tuple[str, List[str]]] = classifier.predict_set(data.get_testing_data())
    for pred in predicted:
        print("Predicted class: {}, instance: {}".format(pred[0], pred[1]))
