import csv

from typing import List

def preprocess(filename: str) -> List[List[str]]:
    data: List[List[str]] = list()
    with open(filename) as csvfile:
        for row in csv.reader(csvfile):
            data.append(row)
    csvfile.close()
    return data