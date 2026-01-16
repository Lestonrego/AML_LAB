#generating overlapping data

import matplotlib.pyplot as plt
from data import overlappingdata
X,y=overlappingdata()
plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr')
plt.title("Overlapping data")
plt.show()