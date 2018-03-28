from numpy import random
MAX_ITERATE: int = 10
'''
def classes_distro(prob, n):
	a = random.sample(range(1, n), n) + [0, n]
	list.sort(a)
	return ([a[i+1] - a[i] for i in range(len(a) - 1)])

print(classes_distro(10, 4))
'''

#---------------------------------------------------------------
import numpy as np
from typing import List
#print(np.random.dirichlet(np.ones(4), size = 1))

def generate_class_prob(n) -> List[float]:
    """
    Generate the random distribution (non-uniform) for each instance in the classifier
    """
    ls: List[float] = random.dirichlet(np.ones(n), size = 1)
    return ls



def class_prob_matrix(num_instances: int, num_class: int) -> List[List[float]]:
    """
    Generate the list of lists of random distribution for each matching instance 
    """
    prob_matrix = []
    for i in range(num_instances):
    	# Generate a list of random distribution per instance on the data table
        prob_matrix.append(generate_class_prob(num_class))

    return prob_matrix

#print(class_prob_matrix(5, 2))

###############################################################################
'''
def predict_iterate(self, data: ClassifierData):
	# Each iteration
	for iteration in range(MAX_ITERATE):
		distro_list = []
		#self.
'''


###############################################################################
# 





# Iterate a number of times to train the data set and get different sets of class distributions
def predict_iterate2(raw_data: List[List[str]], class_distribution: List[List[float]]):
	distro_list = []
	# For every class distribution normalise the distributions
	for iteration in range(MAX_ITERATE):
		#train data and receive the class_distros for each iteration
		#train(raw_data)
		# For every instance normalise the probability generated
		single_distro = []
		for instance in class_distribution:
			single_distro.append(normalise(class_distribution))
		# Add to the final output
		distro_list.append(single_distro)
	return distro_list

# Normalising the probabilities
def normalise(ls: List[float]) -> List[float]:
	return [float(i)/sum(ls) for i in ls]


ls = class_prob_matrix(10, 4)
str_ls = [['1'],['2']]

print(predict_iterate2(str_ls,ls))