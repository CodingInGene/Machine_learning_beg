#Make multiple linear regression model for housing prices
#x1-> total rooms, x2->median income, y->median house value
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def getB(x,y):
    b=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.transpose(x)),y)    #B shape will be same as y
    return b

def predict(x,b):   #Send x as matrix rows from X -> [1,x1,x2], send b as matrix -> [b0,b1,b2]
    #res=b[0]+(b[1]*x[1])+(b[2]*x[2])
    n=len(b)	#How many independent vars
    
    res=b[0]
    for i in range(1,n):	#Make equation for n indep. vars
        res=(b[i]*x[i])+res
    
    return res

def getRsquared(ytest,ycap):
    meanYtest=np.mean(ytest)    #Mean of the response variable
    print("Mean of Y test",meanYtest)

    #SSR sum of squares of regression - (ypredicted-mean)^2
    ssr=0
    for i in ycap:
        ssr=((i-meanYtest)**2)+ssr
    print("SSR",ssr)
    
    #SST sum of squared total - (yactual-mean)^2
    sst=0
    for i in ytest:
        sst=((i-meanYtest)**2)+sst
    print("SST",sst)

    #R squared or R2
    r2=(ssr/sst)
    return r2

if __name__=="__main__":
    housing=pd.read_csv("housing.csv")
    housing.dropna(how='any',inplace=True)
    print(housing)

    #Whole data set
    x1=[]   #total rooms
    x2=[]   #Income
    x3=[]   #housing_median_age
    Y=[]

    for i in housing["total_rooms"]:
        x1.append(i)

    for i in housing["median_income"]:
        x2.append(i)

    for i in housing["housing_median_age"]: # housing_median_age increases the performance to 71% from 43%
        x3.append(i)

    for i in housing["median_house_value"]:
        Y.append(i)

    #Create dep. and indep. vars
    X=np.transpose([np.ones(len(x1)),x1,x2,x3])
    
    #Training set
    b=getB(X[:10000],Y[:10000])  #Get b0 - bn   ; training set
    print(b)

    #Test data
    ycap=[]     #Prediction data
    for i in X[15001:18000]:          #Taking 4999 data for test from 20433 rows after training
        ycap.append(predict(i,b))

    #Get R squared
    r2=getRsquared(Y[15001:18000],ycap)
    print("R2 score (Coefficient of determination) is:",r2)
    print("Performance: "+str(round(r2*100))+" %")

    #Sklearn r2 score
    sk_r2=r2_score(Y[15001:18000],ycap)
    print("r2 score(sklearn)",sk_r2)

    
    
    #plot
    newX=np.transpose(X[15001:18000])

    #ax=plt.axes(projection='3d')
    plt.style.use("ggplot")
    #ax.scatter(newX[2],newX[1],Y[2001:7000],marker='o',facecolor='grey',alpha=0.5,edgecolor='k')   #Actual
    #ax.scatter(newX[2],newX[1],ycap,facecolor="b",marker='o',alpha=0.7)    #Predict

    fig,(ax1,ax2)=plt.subplots(2)

    #1st plot
    ax1.scatter(newX[2],Y[15001:18000],marker='o',facecolor='grey',alpha=0.5,edgecolor='k')   #Actual
    ax1.scatter(newX[2],ycap,facecolor="b",marker='o',alpha=0.7)    #Predict
    ax1.set_xlabel("Median income")
    ax2.set_ylabel("y predicted")

    #2nd plot
    ax2.scatter(newX[1],Y[15001:18000],marker='o',facecolor='grey',alpha=0.5,edgecolor='k')   #Actual
    ax2.scatter(newX[1],ycap,facecolor="b",marker='o',alpha=0.7)    #Predict
    ax2.set_xlabel("Total rooms")
    ax2.set_ylabel("y predicted")
    
    fig.suptitle(f"Multiple linear regression analysis (Training data - 10000). Efficiency: {round(r2*100)}%")
    ax1.legend(["Actual","Predict"])
    ax2.legend(["Actual","Predict"])
    plt.show()
    
    # fig=go.Figure()
    # fig.add_trace(go.Scatter3d(x=newX[2],y=newX[1],z=Y[2001:7000],mode='markers',marker=dict(size=5)))
    # fig.add_trace(go.Scatter3d(x=newX[2],y=newX[1],z=ycap,mode='markers',marker=dict(size=5)))
    # #fig.add_trace(go.Surface(z=np.transpose([newX[1],newX[2],ycap])))
    # fig.show()
