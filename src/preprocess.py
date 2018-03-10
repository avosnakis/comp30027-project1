import csv

from ClassifierData import ClassifierData

from typing import List, Tuple


def preprocess(filename: str) -> ClassifierData:
    return ClassifierData(filename)