import csv

from typing import List, Tuple

TRAINING_FRACTION = .8

def preprocess(filename: str) -> Tuple[List[List[str]], List[List[str]]]:
    training_data: List[List[str]] = list()
    testing_data: List[List[str]] = list()
    num_instances = _get_num_instances(filename)
    num_training_instances: int = round(TRAINING_FRACTION * num_instances)

    with open(filename) as csvfile:
        i: int = 0
        for row in csv.reader(csvfile):
            if i < num_training_instances:
                training_data.append(row)
            else:
                testing_data.append(row)
            i += 1

    csvfile.close()
    return (training_data, testing_data)

def _get_num_instances(filename : str) -> int:
    with open(filename) as csvfile:
        return sum(1 for row in csvfile)
