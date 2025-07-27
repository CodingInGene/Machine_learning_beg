#Predict house value on small inputs
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt

def getB(x,y):
    xt=np.transpose(x)
    b=np.dot(np.dot(np.linalg.inv(np.dot(xt,x)),xt),y)  #Get 1x3 matrix [b0,b1,b2]
    return b

def predict(x1,x2,b0,b1,b2):
    y=b0+(b1*x1)+(b2*x2)
    return y

if __name__=="__main__":
    #Training set
    size=[1,2,3]
    rooms=[3,4,7]
    price=[2,5,9]   #y

    #Create X
    XT=[[1,1,1],size,rooms]
    X=np.transpose(XT)

    #Create Y
    Y=np.transpose(price)   #1x3 matrix of Y to get output as 1d array
    print(Y)
    # Y=[]
    # for i in price:
    #     Y.append([i])

    b0,b1,b2=getB(X,Y)
    print(b0,b1,b2)

    #Real set
    x1=[4,5,6,8,9,12,14,15,16,18,19,20]
    x2=[8,9,10,13,14,15,17,19,20,22,23,24]
    yactual=[10,15,18,23,27,34,42,44,48,52,57,61]
    ypredict=[]
    for i in range(0,len(x1)):
        ycap=predict(x1[i],x2[i],b0,b1,b2)
        ypredict.append(ycap)
        print(ycap,"crores")

    #Plot
    #plt.scatter(x=x1,y=yactual,color="g",marker='.')
    plt.scatter(x=x2,y=yactual,color="g",marker='.')
    #plt.scatter(x1,ypredict,color="r",marker='.')
    plt.plot(x2,ypredict,color="r",marker='.')
    plt.show()