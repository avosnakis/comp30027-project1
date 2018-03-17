import random

def generate_values(n):
    lst = []
    probs = 10

    for i in range(n):
    	
        value = random.randint(1,n)
        lst.append(value/10)
        probs -= value

    return lst 

print(generate_values(4))