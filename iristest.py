# Rebecca Turley, 2019-03-30
# Fisher's Iris Dataset

import numpy as np

data=np.genfromtxt('iris.csv', delimiter=',')

columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']


firstcol = data[:,0]
meanfirstool = np.mean(data[:,0])
print ("Average of first column is:", meanfirstool)

seconcol = data[:,1]
meansecontool = np.mean(data[:,1])
print ("Average of second column is:", meansecontool)

thrdcol = data[:,2]
meanthrdtool = np.mean(data[:,2])
print ("Average of third column is:", meanthrdtool)

fourcol = data[:,3]
meanfourtool = np.mean(data[:,3])
print ("Average of fourth column is:", meanfourtool)

print (firstcol)

print (np.min(firstcol))

print (np.max(firstcol))

print (seconcol)

print (np.min(seconcol))

print (np.max(seconcol))

print (thrdcol)

print (np.min(thrdcol))

print (np.max(thrdcol))

print (fourcol)

print (np.min(fourcol))

print (np.max(fourcol))

import matplotlib.pyplot as pl

pl.hist(firstcol)
pl.title ("Sepal Lenght")
pl.show ()
# histogram of each input variable to get an idea of the distribution
pl.hist(seconcol)
pl.title ("Sepal Width")
pl.show ()
# histogram of each input variable to get an idea of the distribution
pl.hist(thrdcol)
pl.title ("Petal Lenght")
pl.show ()
# histogram of each input variable to get an idea of the distribution
pl.hist(fourcol)
pl.title ("Sepal Width")
pl.show ()
# histogram of each input variable to get an idea of the distribution

print (data.shape)
#to look at the data ie. how many lines and columns

import seaborn as sns
data = sns.load_dataset("iris")
print(data.head())
#load the iris data from the seaborne’s builtin dataset  and print first 5 rows

print(data.describe())
#print some summary statistics

(data['species'].unique())
print(data.groupby('species').size())
#names of the iris and how many of each



#Boxplots
#The boxplot is a quick way of visually summarizing one or more groups of numerical data through their quartiles. Comparing the distributions of:

#Sepal Length
#Sepal Width
#Petal Length
#Petal Width


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="cubehelix", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Sepal Length"

sns.boxplot(x="species", y="sepal_length", data=data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="cubehelix", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Sepal Width"

sns.boxplot(x="species", y="sepal_width", data=data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="cubehelix", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Petal Length"

sns.boxplot(x="species", y="petal_length", data=data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()

#Looking at the boxplots, it becomes apparent that there are large variations in the differences in all four catogeries.  This would indicate that it should be easier to differentiate between the species based on the width and lenghts.



import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="cubehelix", rc={'figure.figsize':(11.7,8.27)})

title="Compare the distributions of Petal Width"

sns.boxplot(x="species", y="petal_width", data=data)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()







import matplotlib.pyplot as plt
from sklearn import datasets
iris= datasets.load_iris()

fig, axes = plt.subplots(nrows= 2, ncols=2)
colors= ['orange', 'pink', 'purple']

for i, ax in enumerate(axes.flat):
    for label, color in zip(range(len(iris.target_names)), colors):
        ax.hist(iris.data[iris.target==label, i], label=             
                            iris.target_names[label], color=color)
        ax.set_xlabel(iris.feature_names[i])  
        ax.legend(loc='upper right')


plt.show()

import pandas as pd
from pandas.plotting import parallel_coordinates


data = pd.read_csv('iris2.csv', delimiter=',')
parallel_coordinates(data, 'Name' )
plt.show()
#The use of Parallel Coordinates to view all the data from the 4 categories to give a quick visual.  I created another csv file with a slightly amended name as this iris2 file had headings which the other one didnt.  


#Scatterplots
#Variables are used to show that there is a noticeable difference in sizes between the species. Firstly, we look at the Sepal lenght and Sepal width across the species. 
#The iris Setosa has a significantly smaller sepal width and sepal length than the other two species. This difference repeats for the Petal width and Petal length. The Iris Viginica is the largest species in both.

import matplotlib.pyplot as plt
import seaborn as sns
iris = sns.load_dataset("iris")

ratio = iris["sepal_length"]/iris["sepal_width"]

for name, group in iris.groupby("species"):
    plt.scatter(group.index, ratio[group.index], label=name)

plt.legend()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
iris = sns.load_dataset("iris")

ratio = iris["petal_length"]/iris["petal_width"]

for name, group in iris.groupby("species"):
    plt.scatter(group.index, ratio[group.index], label=name)

plt.legend()
plt.show()

#sckit learn
# The iris data set already exisits in sklearn so I'm going to reuse it

from sklearn.datasets import load_iris
iris = load_iris()

from matplotlib import pyplot as plt

# The indices of the features that we are plotting
x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()


print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDClassifier

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could
# avoid this ugly slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
colors = "bry"

# shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

h = .02  # step size in the mesh

clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('tight')

# Plot also the training points
for i, color in zip(clf.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                cmap=plt.cm.Paired, edgecolor='black', s=20)
plt.title("Decision surface of multi-class SGD")
plt.axis('tight')

# Plot the three one-against-all classifiers
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = clf.coef_
intercept = clf.intercept_


def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([xmin, xmax], [line(xmin), line(xmax)],
             ls="--", color=color)


for i, color in zip(clf.classes_, colors):
    plot_hyperplane(i, color)
plt.legend()
plt.show()

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("Accuracy Scores:")
print(accuracy_score(Y_validation, predictions))
print("Confusion Matrix:")
print(confusion_matrix(Y_validation, predictions))
print("Classification Report:")
print(classification_report(Y_validation, predictions))


# References:

# https://www.shanelynn.ie/python-pandas-read_csv-load-data-from-csv-files/, http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
# http://statweb.stanford.edu/~jtaylo/courses/stats202/visualization.html, http://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
# https://matplotlib.org/users/pyplot_tutorial.html, https://matplotlib.org/users/pyplot_tutorial.html, https://matplotlib.org/gallery/subplots_axes_and_figures/demo_tight_layout.html
# https://seaborn.pydata.org/tutorial/color_palettes.html?highlight=palette, https://stackoverflow.com/questions/45862223/use-different-colors-in-scatterplot-for-iris-dataset