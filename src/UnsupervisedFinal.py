"""
Machine Learning - COMP30027
Project 1 Code for Unsupervised Naive Bayes Classification
Written by 828472
"""
#===================================================================================
# Libary Imports and Global Definitions
import numpy as np
import csv
from typing import List, DefaultDict, Dict, Callable, Tuple
from collections import defaultdict
from functools import reduce
from pprint import pprint
from numpy import random



EMPTY_CELL: str = "?"
CLASS_CELL: int = -1
EPSILON: float = 10e-7
MAX_ITERATE: int = 10
FRACTION: float = 0.8

#===================================================================================
# Declared Data Structures
"""
Definitions of Terms:
    file       - File string for path and filename of csv
    Data Table - Original Data File Read into the program from a CSV used for training data
    test_data  - Data from the original data_frame sliced off and left for testing
    Classifier - Modified Data Structure which takes the form of a List of double
                 nested dictionaries of floats
    num_instances - Number of instances in the raw data table
    num_class  - Number of classes in the raw data table
    classes_list - List containing all the types of classes
    class_prob_matrix - randomly (non-uniform) class probabilities per instance in raw data table (raw data stage)
    norm_class_distributions -  Matrix of each normalised class distributions per instance for each iteration (prediction)
"""
file: str = '../2018S1-proj1_data/car.csv'
data_table: List[List[str]] = list()
test_data: List[List[str]] = list()
classifier: List[DefaultDict[str, DefaultDict[str, int]]] = list()

num_instances: int = 0
num_class: int = 0
classes_list: List[str] = list()
class_prob_matrix: List[List[float]] = list()
norm_class_distributions: List[List[List[float]]] = list()

#===================================================================================
# File Reading

# FINALISED
def construct_data_table(file: str) -> None:
    """
    Reads the any csv files and converts them into training and test data of 2d lists of strings
    :param file: String containing the directory of the csv file read
    """
    outer_ls: List[List[str]] = list()

    # Open the file
    with open(file) as fr:
        data_frame = csv.reader(fr)

        # Loop through every row and append them into the outer list
        for row in data_frame:
            outer_ls.append(row)

    # Close csv file and shuffle rows
    fr.close()
    random.shuffle(outer_ls)

    # Process the raw unsplit_table called outer_ls
    process_table(outer_ls)

    return None

# FINALISED
def process_table(unsplit_table: List[List[str]]) -> None: 
    """
    Obtains the number of instances, classes, classes_list, and training data
    Splits the raw data taken as a param into training (data_table) and test (test_data)
    :param unsplit_table: 2d list of original but shuffled file read from the csv file
    """
    
    # Get the number of total instances in unsplit_table
    num_instances = len(unsplit_table)
    # Calculate the number of instances for the training instances
    num_training_instances: int = round(FRACTION * num_instances)

    # Loop through the unsplit table and split them into training and test
    i: int = 0
    for row in unsplit_table:
        # Append the classes names into classes_list if not alread in it
        if row[CLASS_CELL] not in classes_list:
            classes_list.append(row[CLASS_CELL])

        # Append the training instances (rows) to data_table
        if i < num_training_instances:
            data_table.append(row)
        # Else, append the row to test_data
        else:
            test_data.append(row)
        i += 1

    # Get the number of classes
    num_class = len(classes_list)

    return None


# Initialising data_table
construct_data_table(file)
#===================================================================================
# Training

# FINALISED
def generate_class_prob(num_class: int) -> List[float]:
    """
    Generate the random distribution (non-uniform) for each instance in the classifier
    :param num_class: The number of classes in the dataset
    :return: The randomised generated probabilities of classes per instance in the data table
    """
    ls: List[float] = random.dirichlet(np.ones(num_class), size=1)
    return ls


# FINALISED
def class_prob_matrix(num_instances: int, num_class: int) -> List[List[float]]:
    """
    Generate the list of lists of random distribution for each matching instance 
    :param num_instances: The number of instances in the original data table
    :param num_class: The number of classes in the dataset
    :return: The list of lists of generated probabilities of a data table
    """
    prob_matrix = []
    for i in range(num_instances):
        # Generate a list of random distribution per instance on the data table
        prob_matrix.append(generate_class_prob(num_class))

    return prob_matrix


#===================================================================================
# Prediction

# FINALISED
def normalise(ls: List[float]) -> List[float]:
    """
    Function to normalise the produced class predictions per instance
    :param ls: List of floats pertaining to each instance of the posterior classes
    :returnd: List of the probability floats normalised to a
    dd to 1
    """
    return [float(i)/sum(ls) for i in ls]

# FINALISED
def predict_set(instances: List[List[str]]) -> List[Tuple[str, List[str]]]:
    """
    Predicts the classes for a set of data_table.
    :param instances: A set of instances to classify each data_table.
    :return: A set of instances along with their classification.
    """
    return [(predict(instance), instance) for instance in instances]


def predict(classifier, instance: List[str]) -> str:
    """
    Predicts the class for an instance.
    :param classifier: Overall classifier trained
    :return: String containing predicted class for this instance
    """
    # if the instance has the class still there, slice it off 
    # for the sake of making a prediction
    # n being the number of column, not attributes.
    n_columns: int = len(classifier) 
    if len(instance) == n_columns:
        instance = instance[:CLASS_CELL]


    classes: List[str] = list(self._class_probs.keys())
    probs: List[float] = [self._prob_of(i_class, instance) for i_class in classes]
    max_index: int = probs.index(max(probs))
    return classes[max_index]


#===================================================================================
# Evaluation


#===================================================================================
# Debug/ Print Calls

# Pre-processed table
print(data_table)

# Classifier