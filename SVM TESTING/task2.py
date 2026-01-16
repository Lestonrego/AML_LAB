#margins for overlapping data
from sklearn.svm import SVC
from data import overlappingdata
import numpy as np
from visual import margin
#create svm with overlapping data hard margin 

X,y=overlappingdata()
svm_hard=SVC(kernel='linear',C=1e6)
svm_hard.fit(X,y)
w=svm_hard.coef_[0] #weight bias
b=svm_hard.intercept_[0] #Bias

x_val=np.linspace(X[:,0].min()-1,X[:,0].max()-1,200)
y_decision = -(w[0]*x_val+b)/w[1]

y_margin_pos=-(w[0] * x_val+b-1)/w[1]
y_margin_neg=-(w[0] * x_val+b+1)/w[1]
print(" W ",w)
print(" b ",b)
print("number of support vectors",len(svm_hard.support_vectors_))
margin(X,y,svm_hard,x_val,y_decision,y_margin_neg,y_margin_pos)