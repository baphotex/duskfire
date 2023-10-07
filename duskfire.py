"""duskfire 
all the helper functions
- nyx  : linear algebra 
- flame: sklearn
- mist : matplotlib
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LogisticRegression

"""
nyx
linear algebra helper functions
"""
least_squares = lambda X,y: np.linalg.inv(X.T@X)@(X.T@y) 

"""
flame
sk learn helper functions
"""
def lr(X,y):
    """logistic regression
    
    parameters
    ----------
    X: numpy array
        feature X
    y: numpy 1d array or pd series
        series containing 0/1s

    returns 
    -------
    tuple
        a tuple of:
        - the model 
        - score
        - coefficients
        - intercept
    """
    model = LogisticRegression(C=1000) 

    print(f"Fitting a Logistic Regression on y vs X")
    X,y = np.array(X), np.array(y) 
    model.fit(X,y)

    score = model.score(X,y)
    print(f"Training accuracy of {np.round(score,4)*100}% on X and y.")
    w,b = model.coef_, model.intercept_[0]
    
    return (model, score, w,b)


def lr_model_plot(X,y,byo=False,model=None,labels=None):
    """logistic regression
    
    parameters
    ----------
    X: numpy array
        feature X, must be essentially 1d
    y: numpy 1d array or pd series
        series containing 0/1s
    model: tuple, optional
        a tuple of form (model,score,w,b) as outputed 
        in lr
        default None
    labels: tuple, optional
        of form (xlab,ylab,title)
        default None

    returns 
    -------
    tuple (tuple,tuple)
        a tuple of:
        - fig 
        - ax
        a tuple of:
        - the model 
        - score
        - coefficients
        - intercept
    """

    model = model if model else lr(X,y)

    x = X.squeeze() 

    xlabel,ylabel,title=labels if labels else ("x-axis","y-axis","Logistic Regression on y-vs-x")

    (m,score,w,b) = model
    w = w[0][0] 
    
    fig, ax = plt.subplots()
    plt.scatter(x,y, s=16,color='black', label=r"data $\{(x_i,y_i)\}$")

    xplot = np.linspace(start=x.min(), stop=x.max()).reshape(-1,1)
    yplot = 1 / (1 + np.exp(-(w * xplot + b)))
    plt.plot(xplot, yplot, label=r'logistic curve $\hat{P}(y = 1)$')

    x_values, x_counts = np.unique(X, return_counts=True)
    n_x_values = x_values.shape[0]
    Xarr,yarr =np.array(X), np.array(y)
    success_proportion_per_x_value = [np.sum(yarr[Xarr[:, 0] == x_values[i]]) \
                            / x_counts[i] for i in np.arange(n_x_values)]

    plt.plot(x_values, success_proportion_per_x_value, '.', color='red',
         label='sample proportions')

    ax.set(xlabel=xlabel, ylabel=ylabel,title=title)
    ax.set_xlim(x.min(), x.max())

    ax.grid(True, which='both')
    #ax.axhline(y=0, color='k')
    #ax.axvline(x=0, color='k')

    plt.legend()
    plt.show()

    return ((fig,ax), (model))

def train_tree(X,y,md):
    """decision tree training function
    
    parameters
    ----------
    X: numpy array
        feature X, must be essentially 1d
    y: numpy 1d array or pd series
        series containing 0/1s
    md: int
        integer>0, max depth of the tree

    returns 
    -------
    decision tree model
    """
    clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=md) 
    clf.fit(X,y)

    score = clf.score(X,y)
    maxdepth = clf.tree_.max_depth
    print(f"\nTraining the decision tree on max depth {md} on the Titanic training set, we get the following.",
            "The score of the decision tree on the Titanic training set is:",
            f'clf.score(X, y)={np.round(score,3)*100}%'
            "\nThe max depth of the decision tree on the Titanic training set is:",
            f"max_depth={maxdepth}",sep="\n")

    return clf

"""
mist
plotting helper functions
"""

def plot_as_line(a,b, just=False):
    """plots numpy series on a-b axes
    
    parameters
    ----------
    a: numpy 1d array or pandas series
        the x
    b: numpy 1d array or pandas series
        the f(x) 
    just: boolean, optional
        a flag to determine whether
        to set the y axis limits to
        the x axis limits (default 
        is false)
    hv: boolean, optional
        a flag to determine whether
        to set the y and x axis lines
        (default is false)

    returns 
    -------
    tuple
        a tuple of the (fig,ax) used 
        in the plotting 
    """
    fig, ax = plt.subplots()
    ax.plot(a,b)

    ax.set(xlabel='$x_1$', ylabel='$x_2$',
        title='$x_2 = f(x_1)$.')
    ax.set_ylim(ax.get_xlim()) if just else None

    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    plt.show()

    return (fig,ax)

def plot_points(a,b, line=False, just=False, hv=False):
    """plots numpy series on a-b axes (scatter)
    
    parameters
    ----------
    a: numpy 1d array or pandas series
        the x
    b: numpy 1d array or pandas series
        the f(x) 
    line: boolean, optional
        a flag to determine whether
        to have a line plot through 
        the points
    just: boolean, optional
        a flag to determine whether
        to set the y axis limits to
        the x axis limits (default 
        is false)
    hv: boolean, optional
        a flag to determine whether
        to set the y and x axis lines
        (default is false)

    returns 
    -------
    tuple
        a tuple of the (fig,ax) used 
        in the plotting 
    """
    fig, ax = plt.subplots()
    plt.scatter(a,b) 
    ax.plot(a,b) if line else None

    ax.set(xlabel='$x_1$', ylabel='$x_2$',
        title='$x_2 = f(x_1)$.')
    ax.set_ylim(ax.get_xlim()) if just else None

    ax.grid(True, which='both')
    ax.axhline(y=0, color='k') if hv else None
    ax.axvline(x=0, color='k') if hv else None

    plt.show()

    return (fig,ax)

def plot_3d(X,Y,Z, wire=False):
    """plot in 3d
    
    parameters
    ----------
    X: numpy array(meshgrid)
        with Y, a meshgrid of points
        from say an array
    Y: numpy array (meshgrid)
        with X, a meshgrid of points
        from say an array
    Z: numpy array
        some Z 
    wire: bool, optional
        optional flag to return a 
        wireframe plot instead
        (default False)

    returns 
    -------
    tuple
        a tuple of the (fig,ax) used 
        in the plotting 
    """

    fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X,Y,Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.plot_wireframe(X, Y, Z, color='black') if wire else None
    ax.view_init(-10, 20)
    plt.show()

    return (fig,ax)