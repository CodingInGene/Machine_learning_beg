#Simple linear regression using Stochastic Gradient descent
import numpy as np
import pandas as pd
import random
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

#Function block
def getData():
    income=pd.read_csv("incomeData_SLR.csv")
    x=[]
    for i in income["income"]:
        x.append(i)
    y=[]
    for i in income["happiness"]:
        y.append(i)
    return x,y

class SgdRegressor:
    def __init__(self):
        self.intercept=None
        self.coeff=None
        self.epochs=None
        self.y_pred=None

    def tranTestSplit(self,x,y,splitnum):
        xtrain=x[:round(len(x)*splitnum)]
        ytrain=y[:round(len(y)*splitnum)]
        xtest=x[round(len(x)*splitnum):]
        ytest=y[round(len(y)*splitnum):]

        return xtrain,ytrain,xtest,ytest

    def fit(self,x,y,learning_rate,epochs):
        self.intercept=0
        self.coeff=1
        self.lr=learning_rate
        self.epochs=epochs
        self.n=len(x)
        
        for i in range(0,epochs):
            for j in range(0,len(x)):
                r=random.randrange(0,len(x))
                self.y_hat = self.intercept + (x[r]*self.coeff)
                slope_intercept = (y[r]-self.y_hat) * (-2/self.n)
                slope_coeff = ( (y[r]-self.y_hat)*x[r] ) * (-2/self.n)
                self.intercept -= self.lr*slope_intercept
                self.coeff -= self.lr*slope_intercept

    def predict(self,xtest,ytest):
        self.y_pred = np.dot( xtest,self.coeff ) + self.intercept


#Main block
if __name__=="__main__":
    X,Y = getData()   #gets 1 x n
    
    #Split train test data
    sgd=SgdRegressor()
    xtrain,ytrain,xtest,ytest = sgd.tranTestSplit(X,Y,splitnum=0.5)

    #Train model
    sgd.fit(xtrain,ytrain,learning_rate=0.01,epochs=50)
    print("Model coeff",sgd.coeff,"Intercept",sgd.intercept)

    #Predict
    sgd.predict(xtest,ytest)
    y_pred = sgd.y_pred

    #R2 score
    r2=r2_score(ytest,y_pred)
    print("R2 score",r2)

    #Plot
    plt.scatter(xtest,ytest,label="Test set",alpha=0.8)
    plt.plot(xtest,y_pred,color="orange",linewidth=2,label="Model prediction")
    plt.title(f"S.G.D, epochs={sgd.epochs} lr={sgd.lr} efficiency {round(r2*100)}%")
    plt.legend()
    plt.show()