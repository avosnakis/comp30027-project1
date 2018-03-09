"""
Wrapper class for the value matrix
"""

from typing import List, DefaultDict, Dict
from collections import defaultdict

EMPTY_CELL: str = "?"

class ValueMatrix:
    def __init__(self):
        self.__struct: List[DefaultDict[str, DefaultDict[str, int]]] = list()

    def init_struct(self, row):
        for i in range(len(row) - 1):
            self.__new_dict()

    def add_row(self, row: List[str]) -> None:
        """"""
        instance_class: str = row[-1]

        # Skip the last column which is the class
        for i in range(len(row) - 1):
            self.__incr_cell(instance_class, row[i], i)

    def __new_dict(self) -> None:
        """
        Creates a new defaultdict for this class, and a cell mapping for it
        """
        self.__struct.append(defaultdict(lambda: defaultdict(int)))

    def __incr_cell(self, instance_class: str, attr: str, curr_cell: int) -> None:
        if attr == EMPTY_CELL:
            return        
        self.__struct[curr_cell][instance_class][attr] += 1

    def print_matrix(self):
        print(self.__struct)
