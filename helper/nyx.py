"""nyx linear algebra module
internal module for frequent 
linear algebra functions 
"""
import numpy as np

least_squares = lambda X,y: np.linalg.inv(X.T@X)@(X.T@y) 
