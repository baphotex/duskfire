"""nyx linear algebra module
internal module for frequent 
linear algebra functions 
"""
import numpy as np

least_squares = lambda X,y: np.linalg.inv(X.T@X)@(X.T@y) 

ridge_regress = lambda X,y,c: np.linalg.inv(X.T@X + c*np.identity(len(X.T)))@(X.T@y)

normalize = lambda x: x/np.linalg.norm(x)
c_vecs = lambda X:np.array([X[:,i].reshape(-1,1) for i in range(0,len(X.T))])

getV = lambda X: np.concatenate(normalize(get_vecs(X)),axis=1)

_formU = lambda i,v: (np.arange(len(X))==i).astype(int).reshape(-1,1)*np.linalg.norm(v) 
getU = lambda X: np.concatenate([_formU(i,v) for (i,v) in enumerate(get_vecs(X))],axis=1)
