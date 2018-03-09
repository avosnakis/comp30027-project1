"""
Wrapper class for the value matrix
"""

import csv

from typing import List, DefaultDict, Dict
from collections import defaultdict

EMPTY_CELL: str = "?"

class ValueMatrix:
    def __init__(self):
        self.__struct: List[DefaultDict[str, DefaultDict[str, int]]] = list()
        self.__maps: Dict[str, int] = dict()

    def add_row(self, row: List[str]) -> None:
        """"""
        instance_class: str = row[-1]

        if instance_class not in self.__maps:
            self.__new_dict(instance_class)

        # Skip the last column which is the class
        for i in range(len(row) - 1):
            self.__incr_cell(instance_class, row[i], row, i)

    def __new_dict(self, instance_class: str) -> None:
        """
        Creates a new defaultdict for this class, and a cell mapping for it
        """
        self.__maps[instance_class] = len(self.__struct)
        self.__struct.append(defaultdict(lambda: defaultdict(int)))

    def __incr_cell(self, instance_class: str, attr: str, row: List[str], ignored_cell: int) -> None:
        if attr == EMPTY_CELL:
            return        
        cell: int = self.__maps.get(instance_class)
        print(cell)
        for i in range(len(row) - 1):
            if i == ignored_cell:
                continue
            self.__struct[cell][attr][row[i]] += 1

    def print_matrix(self):
        print(self.__maps)
        print(self.__struct)