"""
Wrapper class for the value matrix
"""

from typing import List, DefaultDict, Dict
from collections import defaultdict

from ClassifierData import ClassifierData

EMPTY_CELL: str = "?"
CLASS_CELL: int = -1

class SupervisedNBClassifier:
    def __init__(self):
        self.__struct: List[DefaultDict[str, DefaultDict[str, int]]] = list()
        self._probs: List[DefaultDict[str, DefaultDict[str, float]]] = list()
        self._class_probs: Dict[str, float] = dict()
        self._class_instances: DefaultDict[str, int] = defaultdict(int)
        self._num_instances = 0

    def train(self, data: ClassifierData) -> None:
        self._build_freq_struct(data.get_training_data())
        self._build_class_probs(data.get_classes())

    def _build_freq_struct(self, data: List[List[str]]) -> None:
        """
        Populates and builds the frequency struct.
        """
        first_run: bool = True
        self._num_instances = len(data)
        for instance in data:
            if first_run:
                self._init_struct(instance)
                first_run = False
            self._add_row(instance)

    def _build_class_probs(self, classes: List[str]) -> None:
        """
        Builds the class probabilities dictionary.
        """
        for instance_class in classes:
            self._class_probs[instance_class] = self._find_class_prob(instance_class)

    def _find_class_prob(self, instance_class: str) -> float:
        """
        Determines the probability of a single class appearing over the entire dataset.
        """
        return self._class_instances[instance_class] / self._num_instances

    def _init_struct(self, row: List[str]) -> None:
        """
        Initialises the attribute dictionaries.
        """
        for i in range(len(row) - 1):
            self.__new_dict()

    def _add_row(self, row: List[str]) -> None:
        """
        Processes a row and adds its relevant data to the matrix.
        """
        instance_class: str = row[CLASS_CELL]
        self._class_instances[instance_class] += 1

        # Skip the last column which is the class
        for i in range(len(row) - 1):
            self.__incr_cell(instance_class, row[i], i)


    def __new_dict(self) -> None:
        """
        Creates a new defaultdict of defaultdicts of ints for this attribute.
        """
        self.__struct.append(defaultdict(lambda: defaultdict(int)))

    def __incr_cell(self, instance_class: str, attr: str, curr_cell: int) -> None:
        if attr == EMPTY_CELL:
            return        
        self.__struct[curr_cell][instance_class][attr] += 1

    def print_matrix(self) -> None:
        print("__struct: ", self.__struct)
        print("_class_probs: ", self._class_probs)
        print("_class_instances: ", self._class_instances)
        print("_num_instances: ", self._num_instances)