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




# References:

# https://www.shanelynn.ie/python-pandas-read_csv-load-data-from-csv-files/, http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
# http://statweb.stanford.edu/~jtaylo/courses/stats202/visualization.html, http://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
# https://matplotlib.org/users/pyplot_tutorial.html, https://matplotlib.org/users/pyplot_tutorial.html, https://matplotlib.org/gallery/subplots_axes_and_figures/demo_tight_layout.html
# https://seaborn.pydata.org/tutorial/color_palettes.html?highlight=palette, https://stackoverflow.com/questions/45862223/use-different-colors-in-scatterplot-for-iris-dataset