from ClassifierData import ClassifierData
from SupervisedNBClassifier import SupervisedNBClassifier


def train_supervised(data: ClassifierData) -> SupervisedNBClassifier:
    matrix: SupervisedNBClassifier = SupervisedNBClassifier()
    matrix.train(data)
    return matrix
