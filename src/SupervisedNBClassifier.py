"""
Wrapper class for the value matrix
"""

from typing import List, DefaultDict, Dict
from collections import defaultdict
from functools import reduce
from pprint import pprint

from ClassifierData import ClassifierData

EMPTY_CELL: str = "?"
CLASS_CELL: int = -1

class SupervisedNBClassifier:
    def __init__(self):
        self.__struct: List[DefaultDict[str, DefaultDict[str, int]]] = list()
        self._probs: List[DefaultDict[str, DefaultDict[str, float]]] = list()
        self._class_probs: Dict[str, float] = dict()
        self._class_instances: DefaultDict[str, int] = defaultdict(int)
        self._val_instances: List[DefaultDict[str, int]] = list()
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
        self._build_probs(classes)

    def _find_class_prob(self, instance_class: str) -> float:
        """
        Determines the probability of a single class appearing over the entire dataset.
        """
        return self._class_instances[instance_class] / self._num_instances

    def _build_probs(self, classes: List[str]) -> None:
        for i in range(len(self.__struct)):
            for instance_class in classes:
                for key, val in self.__struct[i][instance_class].items():
                    print("items: ", key, val)
                    self._set_prob(i, instance_class, key, val)

    def _set_prob(self, attr: int, instance_class: str, attr_key: str, attr_val: int) -> None:
        total_in_class = sum(self.__struct[attr][instance_class].values())
        print("attr: ", attr)
        print("class: ", instance_class)
        print("attr_key: ", attr_key)
        print("attr_val: ", attr_val)
        print("total: ", total_in_class)
        self._probs[attr][instance_class][attr_key] = attr_val / total_in_class

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
        self._val_instances.append(defaultdict(int))
        self._probs.append(defaultdict(lambda: defaultdict(float)))

    def __incr_cell(self, instance_class: str, attr: str, curr_cell: int) -> None:
        if attr == EMPTY_CELL:
            return
        self._val_instances[curr_cell][attr] += 1
        self.__struct[curr_cell][instance_class][attr] += 1

    def print_matrix(self) -> None:
        print("__struct: ")
        for item in self.__struct:
            pprint(item)
        print("_class_probs: ", self._class_probs)
        print("_class_instances: ", self._class_instances)
        print("_num_instances: ", self._num_instances)
        print("_probs: ")
        for item in self._probs:
            pprint(item)