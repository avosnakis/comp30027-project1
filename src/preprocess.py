import csv

from ValueMatrix import ValueMatrix

def preprocess(filename: str) -> ValueMatrix:
    matrix: ValueMatrix = ValueMatrix()
    with open(filename) as csvfile:
        for row in csv.reader(csvfile):
            matrix.add_row(row)
    csvfile.close()
    return matrix