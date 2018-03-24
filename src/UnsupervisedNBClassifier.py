"""
Wrapper class for the unsupervised value matrix
"""

from typing import List, DefaultDict, Dict, Callable, Tuple
from collections import defaultdict
from functools import reduce
from pprint import pprint
from numpy import random

from ClassifierData import ClassifierData

EMPTY_CELL: str = "?"
CLASS_CELL: int = -1
EPSILON: float = 10e-7
MAX_ITERATE: int = 10


class UnsupervisedNBClassifier:
    def __init__(self, laplace=True):
        """
        __struct: Container for the Classifier
        _probs: Probability of each attribute given each class
        _class_probs: Probability per iteration for each class happening
        _class_instances: 
        _num_instances: Number of instances in the raw data table
        _laplace: Boolean variable to indicate if laplace is used
        _num_class: Number of classes
        _class_prob_matrix: randomly (non-uniform) class probabilities per instance in raw data table
        _norm_class_distributions: Matrix of each normalised class distributions per instance for each iteration
        """
        self.__struct: List[DefaultDict[str, DefaultDict[str, int]]] = list()
        self._probs: List[DefaultDict[str, DefaultDict[str, float]]] = list()
        self._class_probs: Dict[str, float] = dict()
        self._class_instances: DefaultDict[str, int] = defaultdict(int)
        self._num_instances: int = 0
        self._laplace: bool = laplace
        self._num_class: int = 0
        self._class_prob_matrix: List[List[float]] = list()
        self._norm_class_distributions: List[List[List[float]]] = list()

    def train(self, data: ClassifierData) -> None:
        """
        Trains the classifier, building the frequency table and the classes probability
        dictionary.
        """
        self._num_class = data.get_num_classes()
        self._build_freq_struct(data.get_training_data())
        self._build_class_probs(data.get_classes())

    # FINALISED
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
    
    
    #FINALISED
    def _build_freq_struct(self, data: List[List[str]]) -> None:
        """
        Populates and builds the frequency struct.
        """

        first_run: bool = True
        self._num_instances = len(data)

        #Generate the list of the probabilities given the number of classes
        self._class_prob_matrix = self.class_prob_matrix(len(data), data.get_num_classes)

        #Variable to keep track of the instance number
        i: int = 0

        for instance in data:
            if first_run:
                self._init_struct(instance)
                first_run = False
            self._add_row(instance, i)
            i+=1


    #FINALISED
    def _build_class_probs(self, classes: List[str]) -> None:
        """
        Builds the class probabilities dictionary.
        """
        #Iterating through each row to count the probability of each class
        for instance_class in classes:
            # Works the same way as supervised but with fractional counts
            self._set_class_prob(instance_class)
        self._build_probs(classes)



    #NEEDS REVISION
    def _set_class_prob(self, instance_class: str) -> None:
        """
        Determines the probability of a single class appearing over the entire dataset.
        """

        prob: float = self._class_instances[instance_class] / self._num_instances
        self._class_probs[instance_class] = prob
        

    #FINALISED
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
        #Should stay the same by right
        total_in_class = sum(self.__struct[attr][instance_class].values())
        self._probs[attr][instance_class][attr_key] = attr_val / total_in_class


    #FINALISED
    def _init_struct(self, row: List[str]) -> None:
        """
        Initialises the attribute dictionaries.
        """
        for i in range(len(row) - 1):
            self.__new_dict()


    #FINALISED
    #Changed to add the probability of the class instead of counting 
    def _add_row(self, row: List[str], instance_num: int) -> None:
        """
        Processes a row and adds its relevant data to the matrix.
        """
        # Skip the last column which is the class
        for i in range(len(row) - 1): 
            
            self.__incr_cell(row[CLASS_CELL], row[i], i, instance_num)

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


    #FINALISED
    def __incr_cell(self, instance_class: str, attr: str, curr_cell: int, instance_num: int) -> None:
        """
        Increments the value of an attribute given its conditions.
        :param instance_class: The class for having its value incremented.
        :param attr: The attribute value having its value incremented.
        :param curr_cell: The attribute having its value incremented.
        """
        # Don't increment anything if it's an empty cell
        if attr == EMPTY_CELL:
            return
        self.__struct[curr_cell][instance_class][attr] += self._class_prob_matrix[instance_num][instance_class]

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

    #FINALISED
    def generate_class_prob(self) -> List[float]:
        """
        Generate the random distribution (non-uniform) for each instance in the classifier
        """
        ls: List[float] = random.dirichlet(np.ones(self._num_class), size = 1)
        return ls

    #FINALISED
    def class_prob_matrix(num_instances: int, num_class: int) -> List[List[float]]:
        """
        Generate the list of lists of random distribution for each matching instance 
        """
        prob_matrix = []
        for i in range(num_instances):
            # Generate a list of random distribution per instance on the data table
            prob_matrix.append(generate_class_prob(num_class))

        return prob_matrix

    #FINALISED
    def normalise(ls: List[float]) -> List[float]:
        """
        Normalise the probabilities for the predictor so they sum to 1.
        :param ls: The list of unnormalised probabilities for each iteration.
        """
        return [float(i)/sum(ls) for i in ls]


def _correct_prediction(instance: Tuple[str, List[str]]) -> bool:
    """
    :param instance: The instance to check whether the prediction was correct. The first cell
    in the tuple is the predicted class, the second cell contains the instance itself.
    :return: Whether the prediction was correct.
    """
    return instance[0] == instance[1][CLASS_CELL]
