import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
def linearseperable_data():
    X,y=make_blobs(n_samples=100,centers=2,random_state=42, cluster_std=1.0)
    y=np.where(y==0,-1,1)
    return X,y
def overlappingdata():
    X_overlap,y_overlap=make_blobs(n_samples=100,centers=2,random_state=42,cluster_std=3)
    y_overlap=np.where(y_overlap==0,-1,1)
    return X_overlap,y_overlap