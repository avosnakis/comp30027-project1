# COMP30027 Machine Learning Project 1

__Build__

```
./setup.sh
source venv/bin/activate
jupyter notebook
```

Then open the `src/proj1.ipynb` notebook.

__Naive Bayes imlpementation__

Core data structure: 1D array of dictionaries of dictionaries

Each cell of the outermost array corresponds to each attribute.

In the second layer, each key-value pair corresponds to a class.

In the third layer, each key-value pair corresponds to an attribute value and its number of occurrences.