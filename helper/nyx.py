"""nyx linear algebra module
internal module for frequent 
linear algebra functions 
"""
import numpy as np

least_squares = lambda X,y: np.linalg.inv(X.T@X)@(X.T@y) 

get_V = lambda X: np.concatenate(
    [v/np.linalg.norm(v) for v in\
    [X[:,i].reshape(-1,1) for i in range(0,len(X.T))]],
    axis=1
)

get_U = lambda X: np.concatenate(
    [(np.arange(len(X))==i).astype(int).reshape(-1,1)*np.linalg.norm(v)  for (i,v) in\
    enumerate([X[:,i].reshape(-1,1) for i in range(0,len(X.T))])],
    axis=1
)

normalize = lambda x: x/np.linalg.norm(x)

get_c_vecs = lambda X:[X[:,i].reshape(-1,1) for i in range(0,len(X.T))] 

vform = lambda vecs: np.concatenate([normalize(v) for v in vecs],axis=1)
