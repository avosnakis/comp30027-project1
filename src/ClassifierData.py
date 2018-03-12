import csv
import copy

from typing import List

TRAINING_FRACTION: float = .8
CLASS_CELL: int = -1


class ClassifierData:
    """
    Wrapper class that stores the training data, testing data, and all classes
    in the dataset, along with an API to access this data.
    """
    def __init__(self, filename: str):
        """
        Processes the input CSV into training and testing data, and records all
        classes.
        """
        self._training_data: List[List[str]] = list()
        self._testing_data: List[List[str]] = list()
        self._classes: List[str] = list()

        num_instances = _get_num_instances(filename)
        num_training_instances: int = round(TRAINING_FRACTION * num_instances)

        with open(filename) as csvfile:
            i: int = 0
            for row in csv.reader(csvfile):
                if row[CLASS_CELL] not in self._classes:
                    self._classes.append(row[CLASS_CELL])

                if i < num_training_instances:
                    self._training_data.append(row)
                else:
                    self._testing_data.append(row)
                i += 1
        csvfile.close()

    def get_num_classes(self) -> int:
        """
        Returns the number of classes in the dataset.
        """
        return len(self._classes)

    def get_classes(self) -> List[str]:
        """
        Returns a copy of the list of classes.
        """
        return list(self._classes)

    def get_training_data(self) -> List[List[str]]:
        """
        Returns a deep copy of the training data.
        """
        return copy.deepcopy(self._training_data)

    def get_testing_data(self) -> List[List[str]]:
        """
        Returns a deep copy of the testing data.
        """
        return copy.deepcopy(self._testing_data)

    def print_data(self) -> None:
        print("Training data:")
        print(self._training_data)
        print("Testing data:")
        print(self._testing_data)
        print("Classes:")
        print(self._classes)


def _get_num_instances(filename : str) -> int:
    with open(filename) as csvfile:
        num_instances = sum(1 for row in csvfile)
    csvfile.close()
    return num_instances
