"""
Wrapper class for the value matrix
"""

from typing import List, DefaultDict, Dict
from collections import defaultdict

from ClassifierData import ClassifierData

EMPTY_CELL: str = "?"
CLASS_CELL: int = -1

class ValueMatrix:
    def __init__(self):
        self.__struct: List[DefaultDict[str, DefaultDict[str, int]]] = list()
        self._probs: List[DefaultDict[str, DefaultDict[str, float]]] = list()
        self._class_probs: List[float] = list()

    def train(self, data: ClassifierData) -> None:
        self._build_freq_struct(data.get_training_data())

    def _build_freq_struct(self, data: List[List[str]]) -> None:
        first_run: bool = True
        for instance in data.get_training_data():
            if first_run:
                self._init_struct(instance)
                first_run = False
            self._add_row(instance)

    def _init_struct(self, row: List[str]) -> None:
        for i in range(len(row) - 1):
            self.__new_dict()

    def _add_row(self, row: List[str]) -> None:
        """
        Processes a row and adds its relevant data to the matrix.
        """
        instance_class: str = row[CLASS_CELL]

        # Skip the last column which is the class
        for i in range(len(row) - 1):
            self.__incr_cell(instance_class, row[i], i)


    def __new_dict(self) -> None:
        """
        Creates a new defaultdict for this attribute
        """
        self.__struct.append(defaultdict(lambda: defaultdict(int)))

    def __incr_cell(self, instance_class: str, attr: str, curr_cell: int) -> None:
        if attr == EMPTY_CELL:
            return        
        self.__struct[curr_cell][instance_class][attr] += 1

    def print_matrix(self) -> None:
        print(self.__struct)
