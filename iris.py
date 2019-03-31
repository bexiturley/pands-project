# Rebecca Turley, 2019-03-30
# Fisher's Iris Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn 
data=np.genfromtxt('iris.csv', delimiter=',')
data = pd.read_csv("iris.csv")
columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

from sklearn.datasets import load_iris

iris = load_iris()


print(data)

#type(iris)
#print (data [0])
# References:

# https://www.shanelynn.ie/python-pandas-read_csv-load-data-from-csv-files/
