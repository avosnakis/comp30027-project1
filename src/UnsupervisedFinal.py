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
LAPLACE: bool = True

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
    num_training_instances - Number of instances in the training set (data_table)
    num_class  - Number of classes in the raw data table
    classes_list - List containing all the types of classes
    class_prob_matrix - randomly (non-uniform) class probabilities per instance in raw data table (raw data stage)
    total_class_probs - total sum of fractional counts of all classes in the classifier
    norm_class_distributions -  Matrix of each normalised class distributions per instance for each iteration (prediction)
"""
file: str = '../2018S1-proj1_data/car.csv'
data_table: List[List[str]] = list()
test_data: List[List[str]] = list()
classifier: List[DefaultDict[str, DefaultDict[str, int]]] = list()

# Raw Data
num_instances: int 
num_training_instances: int 
num_class: int 
classes_list: List[str] = list()
class_prob_matrix: List[List[float]] = list()
total_class_probs: DefaultDict[str, float] = defaultdict(float)

# Posteriors

posterior_classifier: List[DefaultDict[str, DefaultDict[str, int]]] = list()



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
    global num_instances
    num_instances = len(unsplit_table)

    # Calculate the number of instances for the training instances
    global num_training_instances 
    num_training_instances = round(FRACTION * num_instances)

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
    global num_class
    num_class = len(classes_list)

    return None


# Initialising data_table
construct_data_table(file)

#===================================================================================
# Training

# FINALISED
def generate_class_prob(classes: int) -> List[float]:
    """
    Generate the random distribution (non-uniform) for each instance in the classifier
    :param classes: The number of classes in the dataset
    :return: The randomised generated probabilities of classes per instance in the data table
    """
    # Call on dirichlet randomiser to generate probabilities
    # Convert the numpy array into a regular non-nested list
    ls: List[float] = ((random.dirichlet(np.ones(classes), size=1)).tolist())[0]

    return ls


# FINALISED
def gen_class_prob_matrix(instances: int, classes: int) -> List[List[float]]:
    """
    Generate the list of lists of random distribution for each matching instance 
    :param instances: The number of instances in the data_table
    :param classes: The number of classes in the dataset
    :return: The list of lists of generated probabilities of a data table
    """
    prob_matrix = []
    for i in range(instances):
        # Generate a list of random distribution per instance on the data table
        prob_matrix.append(generate_class_prob(classes))

    return prob_matrix

# 
def train() -> None:
    
    # Create the initial class_prob_matrix based on data_table
    global class_prob_matrix
    class_prob_matrix = gen_class_prob_matrix(num_training_instances, num_class)

    # Fill the classifier with probabilities
    fill_classifier()

    return None

# FINALISED
def fill_classifier() -> None:

    # Indicate that the loop has began and the classifier needs to be initialised
    first_run: bool = True
    
    i: int = 0

    # Look through every instance in data_table (training) and pass in index of row too
    for instance in data_table:
        if first_run:
            # Initialise Classifier 
            init_classifier(len(instance))
            first_run = False

        # Increment values in classifier based on what is in this instance
        fill_row(instance, i)
        i += 1

    return None


# FINALISED
def fill_row(row: List[str], row_index: int) -> None:
    """
    Look through each row in data_table and fill the
    respective fractional probabilities into the classifier
    :param row: List of strings representing each instance/row in data_table (training)
    :param row_index: Corresponding row index used to index probabilities in class_prob_matrix
    """

    # Since there are probabilities of all classes in each 
    # Loop through number of classes and increment the one for the respective row
    global total_class_probs 
    for i in range(len(classes_list)):
        # Increment the probability of the current class
        total_class_probs[(classes_list[i])] = class_prob_matrix[row_index][i]

    # Incrementing the individual cells in the classifier
    # Skip the last column which is the class
    for attr_index in range(len(row) - 1):

        # Each instance/row in data_table has multiple classes probabilities
        for j in range(len(classes_list)):
            # For each attribute in a row, find respective cell 
            # for attribute, given class and attribute given class and increment it
            incr_cell(attr_index, classes_list[j], row[attr_index], row_index, j)

    return None 


# FINALISED
def incr_cell(attr_list_index: int, curr_class: str, attr: str, row_index: int, class_num: int) -> None:
    """
    Increments the value of an attribute given its conditions.
    :param attr_list_index: The index for the outermost list for the corresponding attribute.
    :param curr_class: The string of the current class in the outer dict having its value incremented.
    :param attr: The string of the attribute in inner dict having its value incremented.
    :param row_index: The index of the the row in data_table
    :param class_num: The respective class index with respects to the outer list of the classifier
    """
    # Don't increment anything if it's an empty cell
    if attr == EMPTY_CELL:
        return
    global classifier 
    classifier[attr_list_index][curr_class][attr] += class_prob_matrix[row_index][class_num]




# FINALISED
def init_classifier(columns: int) -> None:
    """
    Initialises the classifier structure.
    :param columns: Number of columns in the data_table including the class column
    """ 
    # Loop through each column for attributes, ignoring the last one since its the class.
    for i in range(columns - 1):
        new_dict()

    return None 

# FINALISED
def new_dict() -> None:
    """
    Creates new defaultdicts of defaultdicts of floats for this attribute in the
    appropriate data structures.
    """
    # Appends a newly initialised dict into the classifier structure
    classifier.append(defaultdict(lambda: defaultdict(default_prob())))

    return None

# FINALISED
def default_prob() -> Callable[[], float]:
    """
    Initialise default value in inner dictionary
    :return: The appropriate lambda function depending on whether the classifier is using
    Laplace probabilistic smoothing or epsilon smoothing.
    """
    return lambda: 0 if LAPLACE else lambda: EPSILON


# Train Data
train()

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
#print(test_data)

# Classifier
#print(num_instances, num_training_instances, num_class)
#print(class_prob_matrix)


print(classifier)