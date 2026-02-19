import matplotlib.pyplot as plt

def margin(X,y,svm_hard,x_vals,y_decision,y_margin_neg,y_margin_pos):
    plt.figure(figsize=(7,6))
    plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',edgecolors='k')
    plt.scatter(svm_hard.support_vectors_[:,0],
            svm_hard.support_vectors_[:,1],
            s=120,facecolors="none",edgecolors="k",linewidths=2,
            label="Support Vectors")
    plt.plot(x_vals,y_decision,'k-',label="Decision Boundry")
    plt.plot(x_vals,y_margin_pos,'k--',label="Margin+1")
    plt.plot(x_vals,y_margin_neg,'k--',label="Margin-1")

    #shade margin area
    plt.fill_between(x_vals,y_margin_neg,y_margin_pos,color='grey',alpha=0.2,label="Margin area")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Hard margin svm with margin visualization")
    plt.legend()
    plt.show()