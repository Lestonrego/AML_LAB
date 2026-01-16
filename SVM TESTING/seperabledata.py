#from task0 -taks2 linearly seprable data
import matplotlib.pyplot as plt
from data import linearseperable_data
X,y=linearseperable_data()
plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr')
plt.title("Linearly Separable Data")
plt.show()