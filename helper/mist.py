"""mist internal plotting functions 
plotting library for internal use
"""

import numpy as np 
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

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