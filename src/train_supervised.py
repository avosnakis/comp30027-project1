from typing import List

from ValueMatrix import ValueMatrix

def train_supervised(data: List[List[str]]) -> ValueMatrix:
    matrix: ValueMatrix = ValueMatrix()
    first_run: bool = True
    for instance in data:
        if first_run:
            matrix.init_struct(instance)
            first_run = False
        matrix.add_row(instance)
    return matrix
