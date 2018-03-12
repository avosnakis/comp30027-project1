"""
Wrapper class for the value matrix
"""

from typing import List, DefaultDict, Dict, Callable, Tuple
from collections import defaultdict
from functools import reduce
from pprint import pprint

from ClassifierData import ClassifierData

EMPTY_CELL: str = "?"
CLASS_CELL: int = -1
EPSILON: float = 10e-7


class SupervisedNBClassifier:
    def __init__(self, laplace=True):
        self.__struct: List[DefaultDict[str, DefaultDict[str, int]]] = list()
        self._probs: List[DefaultDict[str, DefaultDict[str, float]]] = list()
        self._class_probs: Dict[str, float] = dict()
        self._class_instances: DefaultDict[str, int] = defaultdict(int)
        self._val_instances: List[DefaultDict[str, int]] = list()
        self._num_instances: int = 0
        self._laplace: bool = laplace

    def train(self, data: ClassifierData) -> None:
        """
        Trains the classifier, building the frequency table and the class probability
        dictionary.
        """
        self._build_freq_struct(data.get_training_data())
        self._build_class_probs(data.get_classes())

    def predict_set(self, instances: List[List[str]]) -> List[Tuple[str, List[str]]]:
        return [(self.predict(instance), instance) for instance in instances]

    def evaluate(self, instances: List[List[str]]) -> float:
        """
        Returns the percentage of correct evaluations.
        """
        predictions = self.predict_set(instances)
        return 100 * (len(list(filter(lambda x: x[0] == x[1][CLASS_CELL]
                                      , predictions))) / len(predictions))

    def _correct_prediction(self, instance: List[str]) -> bool:
        predicted_class: str = self.predict(instance)
        return predicted_class == instance[CLASS_CELL]

    def predict(self, instance: List[str]) -> str:
        """
        Predicts the class for an instance.
        """
        # if the instance has the class still there, slice it off 
        # for the sake of making a prediction
        n_attributes: int = len(self.__struct)
        if len(instance) == n_attributes:
            instance = instance[:CLASS_CELL]

        classes: List[str] = list(self._class_probs.keys())
        probs: List[float] = [self._prob_of(i_class, instance) for i_class in classes]
        max_index: int = probs.index(max(probs))
        return classes[max_index]

    def _prob_of(self, instance_class: str, instance: List[str]) -> float:
        """
        Determines the probability that an instance is an instance of a specified class.
        """
        class_prob: float = self._class_probs[instance_class]
        prob: float = reduce(lambda x, y: x * y[0][instance_class][y[1]],
                             zip(self._probs, instance), 1)
        return class_prob * prob

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
                    self._set_prob(i, instance_class, key, val)

    def _set_prob(self, attr: int, instance_class: str, attr_key: str, attr_val: int) -> None:
        total_in_class = sum(self.__struct[attr][instance_class].values())
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
        self.__struct.append(defaultdict(lambda: defaultdict(self.__default_count())))
        self._val_instances.append(defaultdict(self.__default_count()))
        self._probs.append(defaultdict(lambda: defaultdict(self.__default_prob())))

    def __default_count(self) -> Callable[[], int]:
        return lambda: 1 if self._laplace else lambda: 0

    def __default_prob(self) -> Callable[[], float]:
        return lambda: 0 if self._laplace else lambda: EPSILON

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
