# Rebecca Turley, 2019-03-30
# Fisher's Iris Dataset


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn 
data=np.genfromtxt('iris.csv', delimiter=',')
data = pd.read_csv('iris.csv', delimiter=',')
columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

#from sklearn.datasets import load_iris

firstcol = data[:,0]
meanfirstool = numpy.mean(data[:,0])
print ("Average of first solumn is:", meanfirstool)
#df = pd.read_csv (r'C:\Users\Rebecca\Desktop\pands-project\iris.csv')
#mean1 = df['sepal_lenght'].mean()   

#iris = load_iris()


print(data)

#type(iris)
#print (data [0])


























# Calculate the mean of each column
#import pandas as pd
#import numpy as np
#data = np.genfromtxt('iris.csv', delimiter=',')
#print(data.head())
#iris.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']
#print (data)




# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#data = numpy.genfromtxt('iris.csv', delimiter=',')
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#print(data.head())


# Load dataset

#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#data = numpy.genfromtxt('iris.csv', columns=iris_dataset["feature_names"])
#dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(20))