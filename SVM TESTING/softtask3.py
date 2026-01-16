from sklearn.svm import SVC
from data import overlappingdata
import numpy as np
from visual import margin
X,y=overlappingdata()
svm_soft=SVC(kernel='linear',C=1.0)
svm_soft.fit(X,y)
print("NUmber of support vectors:",len(svm_soft.support_vectors_))
w=svm_soft.coef_[0] #weight bias
b=svm_soft.intercept_[0] #Bias

x_val=np.linspace(X[:,0].min()-1,X[:,0].max()-1,200)
y_decision = -(w[0]*x_val+b)/w[1]

y_margin_pos=-(w[0]*x_val+b-1)/w[1]
y_margin_neg=-(w[0]*x_val+b+1)/w[1]
print("W ",w)
print(" b",b)

margin(X,y,svm_soft,x_val,y_decision,y_margin_neg,y_margin_pos)