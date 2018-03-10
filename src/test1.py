import random
'''
def classes_distro(prob, n):
	a = random.sample(range(1, n), n) + [0, n]
	list.sort(a)
	return ([a[i+1] - a[i] for i in range(len(a) - 1)])

print(classes_distro(10, 4))
'''


import numpy as np
print(np.random.dirichlet(np.ones(4), size = 1))