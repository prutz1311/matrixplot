# Python Matrix Scatterplot
This module allows graphing matrix scatterplots in Python.
They are used to visualize data and to understand, what method
of analysis to use by projecting data onto all kinds of pairs of dimensions.
It is used mostly in machine learning and statistical inference.
It requires numpy and matplotlib. Your feedback is welcome!
## Usage
In [1]: from sklearn import datasets

In [2]: iris = datasets.load_iris()

In [3]: import matrixplot as mp

In [4]: mp.mplot(iris.data, iris.target, classification=True, filename=None)


In [5]: diabetes = datasets.load_diabetes()

In [6]: mp.mplot(diabetes.data, diabetes.target, classification=False, filename='foo') # we're doing regression, not classification; save to 'foo.png'
