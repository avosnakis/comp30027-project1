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
        """
        :param instances: A set of instances to classify.
        :return: A set of instances along with their classification.
        """
        return [(self.predict(instance), instance) for instance in instances]

    def evaluate(self, instances: List[List[str]]) -> float:
        """
        :param instances: The instances to evaluate this classifier against.
        :return: the percentage of correct evaluations
        """
        predictions = self.predict_set(instances)
        return 100 * (len(list(filter(lambda x: _correct_prediction(x),
                                      predictions))) / len(predictions))

    def predict(self, instance: List[str]) -> str:
        """
        Predicts the class for an instance.
        """
        # if the instance has the class still there, slice it off 
        # for the sake of making a prediction
        n_columns: int = len(self.__struct)
        if len(instance) == n_columns:
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
        prob: float = reduce(lambda x, y: x * self._prob_or_default(*y, instance_class),
                             zip(self._probs, instance), 1)
        return class_prob * prob

    def _prob_or_default(self, 
                         attr_prob: DefaultDict[str, DefaultDict[str, float]],
                         attr_val: str,
                         instance_class: str) -> float:
        """
        Determines the probability of a value given a class and column.
        If the value is empty, then returns the highest value given those conditions.
        """
        # Value imputation: return the highest prob for this condition
        if attr_val == EMPTY_CELL:
            return max(attr_prob[instance_class].values())
        else:
            return attr_prob[instance_class][attr_val]

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
            self._set_class_prob(instance_class)
        self._build_probs(classes)

    def _set_class_prob(self, instance_class: str) -> None:
        """
        Determines the probability of a single class appearing over the entire dataset.
        """
        prob: float = self._class_instances[instance_class] / self._num_instances
        self._class_probs[instance_class] = prob

    def _build_probs(self, classes: List[str]) -> None:
        """
        Determines every posterior probability.
        :param classes: Every possible class for this dataset.
        """
        for i in range(len(self.__struct)):
            for instance_class in classes:
                for key, val in self.__struct[i][instance_class].items():
                    self._set_prob(i, instance_class, key, val)

    def _set_prob(self, attr: int, instance_class: str, attr_key: str, attr_val: int) -> None:
        """
        :param attr: The index of the attribute which will have its probability set.
        :param instance_class: The class to have its probability set.
        :param attr_key: The key of the attribute having its probability set.
        :param attr_val: The number of instances of this attribute given the conditions.
        """
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
        Creates new defaultdicts of defaultdicts of ints OR floats for this attribute in the
        appropriate data structures.
        """
        self.__struct.append(defaultdict(lambda: defaultdict(self.__default_count())))
        self._probs.append(defaultdict(lambda: defaultdict(self.__default_prob())))

    def __default_count(self) -> Callable[[], int]:
        """
        :return: The appropriate lambda function depending on whether the classifier is using
        Laplace probabilistic smoothing or epsilon smoothing.
        """
        return lambda: 1 if self._laplace else lambda: 0

    def __default_prob(self) -> Callable[[], float]:
        """
        :return: The appropriate lambda function depending on whether the classifier is using
        Laplace probabilistic smoothing or epsilon smoothing.
        """
        return lambda: 0 if self._laplace else lambda: EPSILON

    def __incr_cell(self, instance_class: str, attr: str, curr_cell: int) -> None:
        """
        Increments the value of an attribute given its conditions.
        :param instance_class: The class for having its value incremented.
        :param attr: The attribute value having its value incremented.
        :param curr_cell: The attribute having its value incremented.
        """
        # Don't increment anything if it's an empty cell
        if attr == EMPTY_CELL:
            return
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


def _correct_prediction(instance: Tuple[str, List[str]]) -> bool:
    """
    :param instance: The instance to check whether the prediction was correct. The first cell
    in the tuple is the predicted class, the second cell contains the instance itself.
    :return: Whether the prediction was correct.
    """
    return instance[0] == instance[1][CLASS_CELL]
