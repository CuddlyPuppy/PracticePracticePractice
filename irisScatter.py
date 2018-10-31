from sklearn.datasets import load_iris
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd 

iris = load_iris()

irisDF = pd.DataFrame(iris['data'], columns = iris.feature_names)

pd.plotting.scatter_matrix(irisDF, c = iris['target'], figsize = (15,15), marker = 'o', hist_kwds={'bins':20}, s = 60, alpha = 0.8)

