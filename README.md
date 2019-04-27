# pands-project

This contains my project submission 2019 for the module Programming and Scripting at GMIT.

See here for the instructions (https://github.com/ianmcloughlin/project-pands/raw/master/project.pdf)
## How to download this repository

1.  Go to Github.
2.  Click the download button.

##  How to run the code

1. Make sure to have Python installed.
2. Change directory until you are in the correct one.
3. On the command line type Python followed by the name of the file whose script you wish to run.  Make sure to enter .py at the end.


<p align="center">
<img src=https://github.com/bexiturley/pands-project/blob/master/ronald-fisher.jpg width="200" height="250">

Sir Ronald Aylmer Fisher was a statistician who studied in Cambridge.  Throughout his life he published many texts which were significant in the field of statistics.  This included Statistical Methods for Research Workers (1925) which was a handbook for the methods for the design and analysis of experiments. The contributions Fisher made included the development of methods suitable for small samples and the discovery of the precise distributions of many sample statistics. Fisher published “The design of experiments” (1935) and “Statistical tables” (1947). His books had a huge effect on agricultural research as they described the methods for evaluating the results of small sample experiments and so laying out experimental trials as to minimise the disturbances due to heterogeneity of soils and the unavoidable irregularity of biological material. This method was used throughout the world.  In 1930, he published his theories about gene dominance and fitness in The Genetical Theory of Natural Selection. 
In 1936 Fisher authored the paper “The use of multiple measurements in taxonomic problems” as an example of linear discriminant analysis.  The basic premise of LDA is that it tries to maximise the separation of two or more groups of samples.  https://www.youtube.com/watch?v=azXCzI57Yfc

Fisher’s Iris Dataset refers to three species of iris; Iris Setosa, Iris Versicolour and Iris Virginica.  50 random samples of each variation of the Iris were taken.  Measurements of the petals and sepals width and length were recorded.  All the data was looked at to create five columns in this dataset with the following variable names: Sepal length, Sepal width, Petal length, Petal width, and Species.  The first four variables are real measurements made in centimetres. Two of the three species were collected in the Gaspé Peninsula, all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus.  Fisher attempted to determine if it was possible to classify which species a flower belonged to from looking at the lengths and width of the petals and sepals. Here is a link to his actual paper on it http://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf

 :sparkles: It is also known as Andersons dataset as Edgar Anderson collected the data.  :sparkles: 

***
<p align="center">

![flower](https://user-images.githubusercontent.com/47194968/56472909-6e313c00-645c-11e9-80d9-1f4cffd97d85.png)


The dataset contains a set of 150 records under 5 attributes -

1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. Species: 
    - [ ] Iris Setosa 
    - [ ] Iris Versicolour 
    - [ ] Iris Virginica


***

## Libraries used 
The following are the libraries which were used for the project;

NumPy is the fundamental package for scientific computing in Python.

Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications.

Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms.

Pandas is a software library written for the Python programming language for data manipulation and analysis.

***

## Data import
Import the iris.csv using the panda library.

*data=np.genfromtxt('iris.csv', delimiter=',')*
*columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']*


## Quick look at the data
Print out all the information grouped by column and calculate average of each column.

*firstcol = data[:,0]*
*meanfirstool = np.mean(data[:,0])*
*print ("Average of first column is:", meanfirstool)*

*seconcol = data[:,1]*
*meansecontool = np.mean(data[:,1])*
*print ("Average of second column is:", meansecontool)*

*thrdcol = data[:,2]*
*meanthrdtool = np.mean(data[:,2])*
*print ("Average of third column is:", meanthrdtool)*

*fourcol = data[:,3]*
*meanfourtool = np.mean(data[:,3])*
*print ("Average of fourth column is:", meanfourtool)*

Print out the smallest and largest value in each of the 4 columns.
*print (firstcol)*

*print (np.min(firstcol))*

*print (np.max(firstcol))*

*print (seconcol)*

*print (np.min(seconcol))*

*print (np.max(seconcol))*

*print (thrdcol)*

*print (np.min(thrdcol))*

*print (np.max(thrdcol))*

*print (fourcol)*

*print (np.min(fourcol))*

*print (np.max(fourcol))*

***

## More averages of the four columns 
Including how many items there are per column, the mean, minimum and maximum.  I know this has already been partly covered already but I liked to try different ways to view the same information via seaborn.



*import seaborn as sns*
*data = sns.load_dataset("iris")*
*print(data.describe())*


This prints the same information but in a linear format rather than tabular.  Again there is no real need for it but I was curious to see the different ways the information could be viewed.
*summary = data.describe()*
*summary = summary.transpose()*
*print (summary.head())*

***

## Printing the first 5 columns of the information from the seaborne’s builtin dataset 

*data = sns.load_dataset("iris")*
*print(data.head())*

###### you can specific the number to show here (data.head(15))  eg. 15, this prints out the first 15 lines

***

## Looking at the shape of the table, how many lines and columns

*print (data.shape)*

***

## The different names of the Iris and the number of them 

*(data['species'].unique())*
*print(data.groupby('species').size())*


###### The type int64 tells us that python is storing each value within this column as a 64 bit integer.

***

## Histographs
A histogram shows the frequency on the vertical axis and the horizontal axis is another dimension.
First the Sepal Length, Sepal Width, Petal Length and finally Petal Width.  These graphs give only an outline of information.  

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_1.png)

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_2.png)

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_3.png)

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_4.png)

At first glance it appears that there is a wide variance in the sizes of of the petals and sepals and there is no way to distinguish between the different species of Iris.

To get more detail I then reproduced the four histograms but this time I gave each species a different colour to make it easier to differentiate between them.  All 4 were placed on the one page, rather than a page each.  

![Histograph](https://github.com/bexiturley/pands-project/blob/master/Figure_5.png)

Here it becomes more apparent that there is a distinct difference between the three Iris types.

***

## Boxplots
Boxplots are a measure of how well distributed the data in a data set is.  There were four boxplots created; To compare the distributions of Sepal length, 
Sepal Length, Petal Length and Petal Width.

![](https://github.com/bexiturley/pands-project/blob/master/Figure_6.png)


![](https://github.com/bexiturley/pands-project/blob/master/Figure_7.png)

![](https://github.com/bexiturley/pands-project/blob/master/Figure_8.png)

![](https://github.com/bexiturley/pands-project/blob/master/Figure_9.png)

Again it can be seen that each species have distinct properties with relation to the differences in the sepal and petals.  This should make it easier to classify an iris based on the lengths and widths of sepals and petals.

***

## Parallel coordinates

Are a common way of visualizing high-dimensional geometry and analyzing multivariate data.  In a Parallel Coordinates Plot, each variable is given its own axis and all the axes are placed in parallel to each other. Each axis can have a different scale, as each variable works off a different unit of measurement, or all the axes can be normalised to keep all the scales uniform. 
Values are plotted as a series of lines that connected across all the axes. This means that each line is a collection of points placed on each axis, that have all been connected together.

![](https://github.com/bexiturley/pands-project/blob/master/Figure_10.png)

It can be observed that each species (shown in different colors) has a discriminant profile when considering petal length and width, or that Iris Setosa (here in grey) are more homogeneous in regard to petal length (i.e. less of a variance).

***

## Scatterplots

Scatterplots are similar to line graphs in that they use horizontal and vertical axes to plot data points. However, they have a very specific purpose. Scatterplots show how much one variable is affected by another. The relationship between two variables is called their correlation .


![](https://github.com/bexiturley/pands-project/blob/master/Figure_11.png)

![](https://github.com/bexiturley/pands-project/blob/master/Figure_12.png)

It is quite apparent that the points belong to different groups.  Giving each species of Iris a different colour means it is much easier to see the groupings rather than only one colour being used.  This will make separating the classes easier.  The Setosa species is separately clustered from the rest.  While Virginica and Versicolor are also separate, it is not to the same degree.

More scatter plots to create an array of 2d images.

![](https://github.com/bexiturley/pands-project/blob/master/Figure_13.png)



***

##  Scikit Learn
Simple and efficient tools for data mining and data analysis.  It is a free machine learning library provided by Python.  

  

The data already exists within scikitlearn.  



![](https://github.com/bexiturley/pands-project/blob/master/Figure_15.png)

I used https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
to show the graph in a 3D format.  It is way beyond my current ability to plot this myself but I thought it was a fantastic to be able to manipulate the data to produce a 3D graph.  
###### Spin the scatterplot to see more clearly the relationships between the red and green points. This function works when the iris.py script is run in Python.

Following this graph is a 2D representative of it.

![](https://github.com/bexiturley/pands-project/blob/master/Figure_14.png)


To go a bit further with scikit, some have used it to create supervised learning with K-Nearest Neighbours (KNN).  With this the data is split with some of it being used to train the system and the remainder to test it. There are not enough columns to give a perfect score: this is apparent when looking at the scatterplots, they overlap and run into each other.  So any machine-learning approach that gets a perfect score can be regarded as flawed.  The iris data can be used to train and obtain a high accuracy.  Examples of it in use can be viewed here: https://github.com/Msanjayds/Scikit-learn/blob/master/KNN%20on%20Iris%20Datset.ipynb and https://github.com/RitRa/Project2018-iris.  I used a little of KNN at the last script at the end of the project to classify the species based on inputted data.
***

## Decision Surface of multi-class SGD

![](https://github.com/bexiturley/pands-project/blob/master/Figure_17.png)

The dashed lines represent the three OVA classifiers; the background colors show the decision surface induced by the three classifiers.  The decision boundaries are shown with all the points in the training-set.



The core of many machine learning algorithms is optimization.  Optimization algorithms are used by machine learning algorithms to find a good set of model parameters given a training dataset.  The most common optimization algorithm used in machine learning is stochastic gradient descent.
SGD is beneficial when it is not possible to process all the data multiple times because your data is huge.


I put in a script that will classify the species based on the inputted information

![enter](https://user-images.githubusercontent.com/47194968/56499458-5c07da00-64fe-11e9-8c3d-a61365daebc8.PNG)

Results in 

![Prediction](https://user-images.githubusercontent.com/47194968/56499510-9cffee80-64fe-11e9-9715-459f90ba416a.PNG)


## Conclusion

From the earlier graphs it becomes apparent that there is an overall difference between the different species and their lengths.  While a few within the sample are nearer to a species not their own, overall there is a relatively high degree of certainty when classifying an iris.

A real world application for the theory to classify an item using machine learning is https://cloud.google.com/blog/products/gcp/how-a-japanese-cucumber-farmer-is-using-deep-learning-and-tensorflow.  

## Steps taken when attempting this project.
**1.**  *Research by reading up on what the actual data set is and why it is of significance.*

**2.** *Planning on the the different steps needed to import, evaluate and code.*

**3.**  *Writing code.*

**4.**  *Regular saves (one mistake I made was to abandon code I at the time deemed unnecessary which later proved to be beneficial).*

**5.**  *Looking up other sources of information including youtube, other github repositories and lecturer videos.  And using it to learn new ways of coding.* 

**6.**  *Adjusting code to incorporate new ideas I had along the way.*

**7.**  *Coming to a conclusion about what the data means.*

**8.**  *Look up more youtube tutorials on markdown and how to increase my knowledge on use of it.*

**9.**  *Importing the graphs to the mark down file.*

**10.**  *Creating a final draft of the project.*

**11.**  *Realising there was still a lot more to do and re do it.*

**12.**  *Create another final draft.*

**13.**  *Look into real world applications of what the Iris data set shows.*

**14.**  *Conclusion and final thoughts.*



## Final Thoughts
Originally I had no prior knowledge of this data set and was relatively new to the concept of programming.  I researched background information on numerous websites trying to get a grasp as to the scope and use of the dataset.  I then started by importing the data myself and looking at it in various ways.  Different kinds of graphs highlighted different aspects of the relationships between the three species.  I then scripted different graphs to show that there was in fact a difference between each species which made classifying the species of Iris easier. I furthered my understanding on the creation of graphs from youtube tutorials and lecturer videos which explained areas in greater depth when I struggled with a particular concept.  I learned that while it will not always be straight forward to absolutely identify which species a specimen belongs to, it is possible to narrow the options.  It was necessary for me to go back and forth numerous times in an effort to get the code to work and I ended up abandoning some of it, which I later came to regret as it would have proved beneficial at a later stage.  When I was finally happy with how the data looked, I created a markdown file.  This one was much larger and more complex than the one I had completed previously for the problem set.  I found this project frustrating, enjoyable, interesting and very rewarding.  It was a very beneficial learning experience as not only did I start on the path of machine learning, I also realized that there is not only one way to code and there are many ways to work with data to show it in different forms. It is an ongoing process and I often had to return to a problem again and again to fully realise it.  I often found myself thinking about how to amend and improve the data and project. I became more confident with programming towards the end of the scripting.  I began to dabble in the machine learning aspect and adapted code whereby sepal length and width are entered and the computer classifies the species of Iris based on the entered information. I wanted to test to see if the code worked. The outcome was as predicted so thereby a success.  

###### References:
I used the following references when researching and understanding the project:
https://www.shanelynn.ie/python-pandas-read_csv-load-data-from-csv-files/

http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html

http://statweb.stanford.edu/~jtaylo/courses/stats202/visualization.html

http://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html

https://matplotlib.org/users/pyplot_tutorial.html

https://matplotlib.org/users/pyplot_tutorial.html 

https://matplotlib.org/gallery/subplots_axes_and_figures/demo_tight_layout.html

https://seaborn.pydata.org/tutorial/color_palettes.html?highlight=palette

https://stackoverflow.com/questions/45862223/use-different-colors-in-scatterplot-for-iris-dataset

https://www.kaggle.com/benhamner/python-data-visualizations

https://stackoverflow.com/questions/45721083/unable-to-plot-4-histograms-of-iris-dataset-features-using-matplotlib

https://umap-learn.readthedocs.io/en/latest/basic_usage.html

https://guides.github.com/pdfs/markdown-cheatsheet-online.pdf

https://uk.mathworks.com/help/stats/box-plots.html

https://datavizcatalogue.com/methods/parallel_coordinates.html

https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py

https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_iris.html#sphx-glr-auto-examples-linear-model-plot-sgd-iris-py 

https://gist.github.com/uupaa/f77d2bcf4dc7a294d109

https://pythonspot.com/k-nearest-neighbors/

https://www.edureka.co/blog/k-nearest-neighbors-algorithm/

https://www.youtube.com/watch?v=qIaWozjDyPk

https://github.com/

https://github.com/mtthss/clustering-iris-dataset

https://www.kaggle.com/aschakra/decision-tree-classification-for-iris-dataset

https://github.com/Msanjayds/Scikit-learn/blob/master/KNN%20on%20Iris%20Datset.ipynb 

https://github.com/RitRa/Project2018-iris