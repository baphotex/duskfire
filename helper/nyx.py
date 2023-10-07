"""nyx linear algebra module
internal module for frequent 
linear algebra functions 
"""
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from numpy import concatenate as cat

least_squares = lambda X,y: inv(X.T@X)@(X.T@y) 
ridge_regress = lambda X,y,c: inv(X.T@X + c*np.identity(len(X.T)))@(X.T@y)

normalize = lambda x: x/norm(x)
cols = lambda X:np.array([X[:,i].reshape(-1,1) for i in range(0,len(X.T))])

getV = lambda X: cat(normalize(cols(X)),axis=1)
_formU = lambda i,v: (np.arange(len(X))==i).astype(int).reshape(-1,1)*norm(v) 
getU = lambda X: cat([_formU(i,v) for (i,v) in enumerate(gols(X))],axis=1)
