from numpy import random
import numpy as np
from typing import List
from collections import defaultdict

EPSILON: float = 10e-7

def generate_values(n):
    lst = []
    probs = 10

    for i in range(n):
        
        value = random.randint(1,n)
        lst.append(value/10)
        probs -= value

    return lst 

print(generate_values(4))


#######################################################
data = [["vhigh","vhigh","2","4","big","low","unacc"],["vhigh","vhigh","2","4","big","med","unacc"],["vhigh","vhigh","2","4","big","high","unacc"],["vhigh","vhigh","2","more","small","low","unacc"],["vhigh","vhigh","2","more","small","med","unacc"],["vhigh","vhigh","2","more","small","high","unacc"],["vhigh","vhigh","2","more","med","low","unacc"],["vhigh","vhigh","2","more","med","med","unacc"],["vhigh","vhigh","2","more","med","high","unacc"],["vhigh","vhigh","2","more","big","low","unacc"],["vhigh","vhigh","2","more","big","med","unacc"],["vhigh","vhigh","2","more","big","high","unacc"],["vhigh","vhigh","3","2","small","low","unacc"],["vhigh","vhigh","3","2","small","med","unacc"],["vhigh","vhigh","3","2","small","high","unacc"],["vhigh","vhigh","3","2","med","low","unacc"],["vhigh","vhigh","3","2","med","med","unacc"],["vhigh","vhigh","3","2","med","high","unacc"],["vhigh","vhigh","3","2","big","low","unacc"],["vhigh","vhigh","3","2","big","med","unacc"],["vhigh","vhigh","3","2","big","high","unacc"],["vhigh","vhigh","3","4","small","low","unacc"]]


#######################################################

#Generate the struct
def create_struct(rows):

    struct = [defaultdict(lambda: defaultdict(lambda: EPSILON))]
    for i in range(rows):
        append_dict(struct)

    return struct

def append_dict(struct) -> None:    
    struct.append(defaultdict(lambda: defaultdict(lambda: EPSILON)))
        


#Create each row, instance in frequency table 
def add_row(struct, row):

    #Generate the list of the probabilities given the number of classes
    ls = generate_class_prob(2)

    # Skip the last column which is the class
    for i in range(len(row) - 1): 
        # For each attribute, there are possibilities for each class
        for j in range(2):
            __incr_cell(struct, row[-1], row[i], i, ls, j)


def __incr_cell(struct, instance_class: str, attr: str, curr_cell: int, ls: List[float], class_num: int) -> None:
        """
        Increments the value of an attribute given its conditions.
        :param instance_class: The class for having its value incremented.
        :param attr: The attribute value having its value incremented.
        :param curr_cell: The attribute having its value incremented.
        """
        # Don't increment anything if it's an empty cell
        if attr == "?":
            return
        struct[curr_cell][instance_class][attr] += ls[class_num]


# Building freq struct
def build_freq_struct(data):
    first_run: bool = True
    num_instances = (len(data))
    
    for instance in data:
        if first_run:
            struct = create_struct(num_instances)
            first_run = False
        add_row(struct, instance)
    return struct

def generate_class_prob(n) -> List[float]:
    """
    Generate the random distribution (non-uniform) for each instance in the classifier
    """
    ls: List[float] = random.dirichlet(np.ones(n), size = 1)
    return ls

print(build_freq_struct(data))

##################################################################################

def class_prob_matrix(num_instances: int, num_class: int, ls: List[float]) -> List[List[float]]:
    """
    Generate the list of lists of random distribution for each matching instance 
    """
    prob_matrix = []
    for i in num_instances:
        prob_matrix.append(generate_class_prob(num_class))

    return prob_matrix