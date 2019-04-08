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
pl.show ()

pl.hist(seconcol)
pl.show ()

pl.hist(thrdcol)
pl.show ()

pl.hist(fourcol)
pl.show ()

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

summary = data.describe()
summary = summary.transpose()
print (summary.head())


# References:

# https://www.shanelynn.ie/python-pandas-read_csv-load-data-from-csv-files/, http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
