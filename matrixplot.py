"""
Plotter for matrix plots that are used in machine learning
and data mining. It plots all 2-D projections of dataset and
can be used for both classification and regression problems.
There is a facility to save plot into a file if it is too large.
"""
import numpy as np
import matplotlib.pyplot as plt


def mplot(data, target, classification=True, filename=None):
    """
    Visualize many-dimensional data to come up with a model.
    
    Parameters
    ----------
    data : ndarray
        Matrix (2-D array) of features, also called independent
        variables (what is called X sometimes), which is passed
        to fit() method of sklearn estimators method as the first argument.
        It should have the shape (n_samples, n_features).
    target : ndarray
        1-D array with labels, also referred to as dependent variables
        and labels (y).It is passed to fit() method of sklearn estimators
        method as the second argument. It should have shape (n_samples,)
    classification : bool
        If this is set to True, labels are interpreted as discrete
        values. Otherwise, colorbar will be contiguous, and
        it will be assumed that we are doing regression. By default,
        it is set to True.
    filename : str
        If given, save the plot as an image 'filename.png' (.png
        will append automatically. If set to None, the plot is displayed
        on the screen. By default, it is set to None.
    """
    count = data.shape[1]
    
    fig = plt.figure(figsize=(count*6,count*4))
    if classification:
        target_range = np.unique(target)
    else:
        target_range = np.linspace(target.min(), target.max(), 65)

    for i in xrange(count):
        for j in xrange (count):
            plt.subplot(count, count, i*count+j+1)
            if i != j:
                plt.scatter(data[:,i],data[:, j],c=target)
            else:
                plt.text(0,0,str(i), fontdict={'size':40})
                plt.colorbar(mappable=plt.contourf(np.vstack(target_range)))
                
    if filename != None:
        plt.savefig(filename + '.png', bbox_inches=0)
    plt.show()
