import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

#Function Block
def getPolyData(n):     #Generate random dummy data, n x 1
    X=6*np.random.rand(n,1)-3
    Y=(0.75*(X**2))+(0.75*X)-0.95+np.random.randn(n,1)
    #Y=(0.75*(X**2))+(0.75*X)-0.95+np.random.randn(n,1)+(0.8*(X**3))+(0.9*(X**4))+(1*(X**5))

    return X,Y

def sigma(item,power):
    sum=0
    for i in item:
        sum=sum+(i**power)
    
    return sum

def sopPoly(x,y,powerX):    #Sum of product of xi, yi
    sum=0
    for i in range(0,len(x)):
        sum=((x[i]**powerX)*y[i])+sum

    return sum

def getB(x,y,deg=2):
    Xcap=np.zeros([deg+1,deg+1])    #n x n
    A=np.zeros([deg+1,1])   #n x 1 matrix

    #X
    Xcap[0][0]=len(x)
    
    for i in range(0,deg+1):
        count=i     #Columnwise power (At each iteration set to i then increase by 1 upto col limit)
        for j in range(0,deg+1):
            if i==0 and j==0:   #Ignoring 00th item
                pass
            else:
                Xcap[i][j]=sigma(x,count)
            count+=1
    
    #A
    A[0][0]=sigma(y,1)
    for i in range(0,deg+1):
        for j in range(0,deg+1):
            if i==0:
                pass
            else:
                A[i][0]=sopPoly(x,y,i)

    B=np.dot(np.linalg.inv(Xcap),A) #intercept and co-effs. Matrix - n x 1
    
    return np.transpose(B)[0]  #1 x n

def predict(x,B):   #Accepts and returns 1 x n
    ycap=[]

    for i in range(0,len(x)):
        sum=B[0]
        for j in range(1,len(B)):
            sum=((x[i]**j)*B[j])+sum
        ycap.append(sum)

    return ycap


#Main Block
if __name__=='__main__':

    n=2000  #Number of data points
    Degree=2    #Degree of polynomial

    #Get the data
    X,Y=getPolyData(n)
    
    #Seperate training and test set
    xtrain=np.transpose(X[:800])[0]       #Convert n x 1 matrix to 1 x n
    ytrain=np.transpose(Y[:800])[0]

    xtest=np.transpose(X[801:])[0]
    ytest=np.transpose(Y[801:])[0]

    # xtrain=[1,2,3,4]  #Pre-tested data
    # ytrain=[1,4,9,15]

    #Get co efficients and intercept
    B=getB(xtrain,ytrain,deg=Degree)     # x, y, degree. #Accepts 1 x n matrix
    print("Intercept",B[0])
    print("Co-effients",B[1:])

    #Prediction
    Ypredict=predict(xtest,B)   #Takes whole X(1 x n) and B(1 x n), returns predicted Y(1 x n)

    #Plot
    plt.scatter(xtrain,ytrain,marker='o',facecolor="lightblue",label="Training set")
    plt.scatter(xtest,ytest,marker='.',facecolor="orange",label="Test set")

    #Smoothing prediction curve
    c=interp1d(xtest,Ypredict,kind="cubic")
    xtest_smooth=np.linspace(xtest.min(),xtest.max(),500)
    ytest_smooth=c(xtest_smooth)

    plt.plot(xtest_smooth,ytest_smooth,'r',label="Prediction")
    plt.legend()
    plt.title(f"Polynomial Regression. (degree-{Degree})")
    plt.ylim(-6,10) #Lower lim -6, upper 10
    plt.show()