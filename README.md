# pands-project

Sir Ronald Aylmer Fisher was a statistician who studied in Cambridge.  Throughout his life he published many texts which were significant in the field of statistics.  This included Statistical Methods for Research Workers (1925) which was a handbook for the methods for the design and analysis of experiments. The contributions Fisher made included the development of methods suitable for small samples and the discovery of the precise distributions of many sample statistics. Fisher published “The design of experiments” (1935) and “Statistical tables” (1947). His books had a huge effect on agricultural research as they described the methods for evaluating the results of small sample experiments and for so laying our experimental trials as to minimise the disturbances due to heterogeneity of soils and the unavoidable irregularity of biological material. This method was used throughout the world.  In 1930, he published his theories about gene dominance and fitness which in The Genetical Theory of Natural Selection. 
In 1936 Fisher authored the paper “The use of multiple measurements in taxonomic problems” as an example of linear discriminant analysis.  The basic premise of LDA is that it tries to maximise the separation of two or more groups of samples.  https://www.youtube.com/watch?v=azXCzI57Yfc

Fisher’s Iris Dataset refers to three species of iris; Iris Setosa, Iris Versicolour and Iris Virginica.  50 random samples of each variation of the Iris were taken with measurements of the petals and sepals both of their width and length.  All the data was looked at to create five columns in this dataset with the following variable names: Sepal.length, Sepal.width, Petal.length, Petal.width, and Species.  The first four variables are real measurements made in centimetres. Two of the three species were collected in the Gaspé Peninsula, all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus.  Fisher attempted to determine if it was possible to classify which species a flower belonged to from looking at the lengths and width of the petals and sepals. Here is a link to his actual paper on it http://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf

It is also known as Andersons dataset as Edgar Anderson collected the data. 

The dataset contains a set of 150 records under 5 attributes -

1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. Species: 
-- Iris Setosa 
-- Iris Versicolour 
-- Iris Virginica

![iris](assets/irises.png)

## Libraries used 
The following are the libraries which were used for the project;

NumPy is the fundamental package for scientific computing in Python.

Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications.

Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms.

pandas is a software library written for the Python programming language for data manipulation and analysis.

## Data Import
Import the iris.csv using the panda library 

data=np.genfromtxt('iris.csv', delimiter=',')
columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

## Quick look at the data
Print out all the information grouped by column and calculate average of each column

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

Print out the smallest and largest value in each of the 4 columns
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

## More averages of the four columns 
Including how many items there are per columns, the mean, minimum and maximum.  I know this has already been partly covered already but I liked to try different ways to view the same information via seaborn.

import seaborn as sns
data = sns.load_dataset("iris")
print(data.describe())

This prints the same information but in a linear format rather than tabular.  Again there is no real need for it but it was my own curiorisity to see the different way to view the information.
summary = data.describe()
summary = summary.transpose()
print (summary.head())


## Printing the first 5 columns of the information from the seaborne’s builtin dataset . 

data = sns.load_dataset("iris")
print(data.head())

#you can specific the number to show here (data.head(15))  eg. 15

## Looking at the shape of the table, how many lines and columns

print (data.shape)

## The different names of the Iris and how many there is 

(data['species'].unique())
print(data.groupby('species').size())
# The type int64 tells us that python is storing each value within this column as a 64 bit integer.


## Histographs
A histogram shows the frequency on the vertical axis and the horizontal axis is another dimension.
First the Sepal Length, Sepal Width, Petal Length and finally Petal Width.  These graphs give only an outline of information.  

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_1.png)

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_2.png)

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_3.png)

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_4.png)

At first glance it appears that there is a wide variance in the sizes of of the petals and sepals and there is no way to distinguish between the different species of Iris.

To get more detail I then reproduced the four histograms but this time I gave each species a different colour to more easier differentiate between them.  And all 4 were placed on the one page, rather than a page each.  

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_5.png)

Here it becomes more apparent that there is a distinct difference between the three Iris.


## Boxplots
Boxplots are a measure of how well distributed the data in a data set is.  There were four boxplots created; To compare the distributions of Sepal length, 
Sepal Length, Petal Length and Petal Width.

![](https://github.com/bexiturley/pands-project/blob/master/Figure_6.png)


![](https://github.com/bexiturley/pands-project/blob/master/Figure_7.png)

![](https://github.com/bexiturley/pands-project/blob/master/Figure_8.png)

![](https://github.com/bexiturley/pands-project/blob/master/Figure_9.png)

Again it can be seen that each species have distinct properties with relation to the differences in the sepal and petals.  This should make it easier to classify an iris based on the lengths and widths of sepals and petals.



## Parallel coordinates. 

Are a common way of visualizing high-dimensional geometry and analyzing multivariate data.  In a Parallel Coordinates Plot, each variable is given its own axis and all the axes are placed in parallel to each other. Each axis can have a different scale, as each variable works off a different unit of measurement, or all the axes can be normalised to keep all the scales uniform. 
Values are plotted as a series of lines that connected across all the axes. This means that each line is a collection of points placed on each axis, that have all been connected together.

![](https://github.com/bexiturley/pands-project/blob/master/Figure_10.png)

It can be observed that each species (shown in different colors) has a discriminant profiles when considering petal length and width, or that Iris Setosa (here in grey) are more homogeneous in regard to petal length (i.e. less of a variance).


## Scatterplots

The scatterplot it’s pretty obvious right away that the points belong to different groups.  it’s much easier to see the groupings than when we just had all blue! We now know that it’ll probably be easy to separate the classes.  The cluster of Setosa species is separately clustered from the rest.  While Virginica and Versicolor are also separate, it is not to the same degree.

![](https://github.com/bexiturley/pands-project/blob/master/Figure_11.png)

![](https://github.com/bexiturley/pands-project/blob/master/Figure_12.png)

More scatter plots to create an array of 2d images.

![](https://github.com/bexiturley/pands-project/blob/master/Figure_13.png)



##  Scikit Learn
Simple and efficient tools for data mining and data analysis.  It is a free machine learning library provided by Python.  From the investigations I undertook during the course of this project
I came to the conclusion that all the above graphs helped to show the differences in the petals and sepals of the iris.  This is how machine learning begins.  Taking all the above data it becomes much 
easier to classify the iris via the differences in lengths and width.  To go one step further is to take the above information see that there is a difference and then put it to use by learning to classify
the iris not by how it looks but by giving the computer measurements and letting it decide for itself which species the iris belongs to. That is machine learning or supervised learning.  The data already exists within sickitlearn.  



![](https://github.com/bexiturley/pands-project/blob/master/Figure_15.png)

I used https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
to show the graph in a 3d format.  It is way beyond my current ability to plot this myself but I thought it was a fantastic to be able to manipulate the data to produce a 3d graph.  Spin the scatterplot to see more clearly the relationships between the red and green points.

Following this graph is a 2d representative of it.

![](https://github.com/bexiturley/pands-project/blob/master/Figure_14.png)

## Decision Surface of multi-class SGD

![](https://github.com/bexiturley/pands-project/blob/master/Figure_17.png)

The dashed lines represent the three OVA classifiers; the background colors show the decision surface induced by the three classifiers.

The final graph 
The core of many machine learning algorithms is optimization.  Optimization algorithms are used by machine learning algorithms to find a good set of model parameters given a training dataset.  The most common optimization algorithm used in machine learning is stochastic gradient descent.
SGD is beneficial when it is not possible to process all the data multiple times because your data is huge.