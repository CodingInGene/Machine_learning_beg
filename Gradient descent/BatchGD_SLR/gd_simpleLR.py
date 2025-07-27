#Apply gradient descent on linear dataset (Simple linear regression)
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

#Function block
def getData():
    incomeData=pd.read_csv("incomeData_SLR.csv")
    #Make x
    X=[]
    for i in incomeData["income"]:
        X.append(i)

    #Make y
    Y=[]
    for i in incomeData["happiness"]:
        Y.append(i)

    return X,Y

def gdSlopes(x,y,b,m):   #Calculate slope with current m,b
    #Slope of b
    sum=0
    for i in range(0,len(x)):
        sum+=(y[i]-(m*x[i]+b))
    slope_b=-2*sum

    #Slope of m
    sum=0
    for i in range(0,len(x)):
        sum+=((y[i]-(m*x[i]+b))*x[i])
    slope_m=-2*sum

    #print("Slope",slope_b,slope_m)

    return slope_b,slope_m

def gdRegressorEpochs(x,y,lr,epochs):     #Implementing epochs approach
    if len(x) is not len(y):
        print("Length of x,y must be same")
        exit()
    count=0
    change_in_b=[]
    change_in_m=[]
    #Take random m,b
    m=0
    b=0
    #Slopes for m = -2 sigma (yi-mxi+b)*xi. for b = -2 sigma (yi-mxi+b)

    #Calculate slopes of m, b
    for i in range(epochs):
        slope_b,slope_m=gdSlopes(x,y,b,m)   #Takes x, y dataset, b, m

        #stepsize
        stepsize_b=lr*slope_b
        stepsize_m=lr*slope_m

        #b, m
        b=b-stepsize_b
        m=m-stepsize_m

        count+=1
        change_in_b.append(b)
        change_in_m.append(m)
    print("Epochs",count)
    
    return b,m,change_in_b,change_in_m

def gdPredict(x,b,m):
    y=[]
    for i in x:
        y.append((m*i)+b)

    return y

def animator(frame):    #Animate the plot of xtrain and y_pred
    animated_plot.set_data(xtest,y_pred_lines[frame])
    return animated_plot,


#Main block
if __name__=="__main__":
    #Get dataset
    X,Y=getData()

    #Split test, training set.  Total rows 498
    xtrain=X[:80]
    ytrain=Y[:80]

    xtest=X[81:]
    ytest=Y[81:]

    #Get coeffs using sklearn
    sk_x=np.array(xtrain).reshape(-1,1)
    sk_y=np.array(ytrain).reshape(-1,1)

    lr=LinearRegression()
    lr.fit(sk_x,sk_y)

    print("Sklearn data - Intercept",lr.intercept_,"Co-effs",lr.coef_)

    #Calculate m,b
    b,m,change_in_b,change_in_m=gdRegressorEpochs(xtrain,ytrain,lr=0.0005,epochs=1000)    #change_in_b,m for animation
    #b,m=gdRegressorStopdiff(xtrain,ytrain)  #Under construction
    print("Model b:",b,"m:",m)

    #Predict
    y_pred=gdPredict(xtest,b,m)

    #Plot the linear graph
    #plt.plot(change_in_b,change_in_m)   #b vs m
    plt.plot(xtest,ytest,'.')
    plt.plot(xtest,y_pred)
    plt.title(f"learning rate - {lr}, Epochs - {epochs}")

    #Plot and animate (Changes in b,m)
    y_pred_lines=[]
    for i in range(0,len(change_in_b)):
        y_pred_lines.append(gdPredict(xtest,change_in_b[i],change_in_m[i]))
    
    fig,axis=plt.subplots()
    axis.plot(xtest,ytest,'.')
    axis.set_xlim([1,8])
    axis.set_ylim([0,7])
    animated_plot,=axis.plot([],[])

    animation=FuncAnimation(
        fig=fig,
        func=animator,
        frames=len(y_pred_lines),
        interval=50,    #millisec
        repeat=True,
    )
    # animation.save("convergence(m=8_b=-8).mp4",writer="ffmpeg",fps=15)
    # print("Animation saved")
    plt.show()