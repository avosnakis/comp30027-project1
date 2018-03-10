from typing import List

from ClassifierData import ClassifierData
from ValueMatrix import ValueMatrix

def train_supervised(data: ClassifierData) -> ValueMatrix:
    matrix: ValueMatrix = ValueMatrix()
    matrix.train(data)
    return matrix
