#Perform polynomial regression using gradient descent
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

#Function block
class GetData:
    def __init__(self,rows):
        self.n=rows
        self.x=6*np.random.rand(self.n,1)-3
        self.y=(0.75*(self.x**2))+(0.75*self.x)-0.95+np.random.randn(self.n,1)

class TrainTestSplit:
    def __init__(self,x,y,splitnum):
        self.n=splitnum
        self.xtrain=x[:round(x.shape[0]*self.n)]
        self.ytrain=y[:round(y.shape[0]*self.n)]

        self.xtest=x[round(x.shape[0]*self.n):]
        self.ytest=y[round(y.shape[0]*self.n):]

class GdRegressor:
    def __init__(self):
        self.intercept=None
        self.coeff=None
        self.y_pred=None

    def fit(self,xtrain,ytrain,degree,learning_rate,epochs):
        self.deg=degree
        self.lr=learning_rate
        self.epochs=epochs

        self.intercept=0
        self.coeff=np.ones([self.deg,1])

        #Prepare x train
        self.x=np.zeros([xtrain.shape[0],self.deg ])    #N x deg
        for i in range(0,self.x.shape[0]):
            for j in range(0,self.x.shape[1]):
                self.x[i][j] = xtrain[i][0]**(j+1)  # xtrain[i][0]^current_col+1

        #Calculate intercept and coeffs
        for i in range(0,epochs):
            y_hat=np.dot(self.x, self.coeff ) + self.intercept
            
            self.slope_intercept = (np.mean(ytrain-y_hat) / self.x.shape[0]) * -2
            self.slope_coeff = (np.dot( np.transpose(self.x),(ytrain-y_hat)) / self.x.shape[0]) * -2
            
            self.intercept = self.intercept - (self.lr*self.slope_intercept)
            self.coeff = self.coeff - (self.lr*self.slope_coeff)

    def predict(self,xtest):
        #Prepare x test
        self.x_new=np.zeros([xtest.shape[0],self.deg ])    #N x deg
        for i in range(0,self.x_new.shape[0]):
            for j in range(0,self.x_new.shape[1]):
                self.x_new[i][j] = xtest[i][0]**(j+1)

        self.y_pred = np.dot(self.x_new, self.coeff) + self.intercept



#Main block
if __name__=="__main__":
    data = GetData(rows=800)
    X=data.x
    Y=data.y

    #Split test and training data
    sp = TrainTestSplit(X,Y,splitnum=0.5)
    xtrain=sp.xtrain
    ytrain=sp.ytrain
    xtest=sp.xtest
    ytest=sp.ytest

    #Train the model
    gd=GdRegressor()
    gd.fit(xtrain,ytrain,degree=2,learning_rate=0.05,epochs=12000)    #Fitting data to regressor

    #Predict
    gd.predict(xtest)
    y_pred = gd.y_pred  #y_pred is working (checked manually)

    print("Model Intercept",gd.intercept,"Coeffs",np.transpose(gd.coeff))

    #Model r2 score
    r2=r2_score(ytest,y_pred)
    print("Model R2 score",r2)

    #Sklearn polynomial prediction
    poly=PolynomialFeatures(degree=2)
    polyXtrain=poly.fit_transform(xtrain)
    polyXtest=poly.fit(xtest)

    lr=LinearRegression()
    lr.fit(polyXtrain,ytrain)

    print("\nSklearn's intercept",lr.intercept_,"Coeff",lr.coef_)

    x_new=np.linspace(-3,3,xtest.shape[0]).reshape(xtest.shape[0],1)
    x_new_trans=poly.transform(x_new)
    y_pred_sk = lr.predict(x_new_trans)

    r2_sk=r2_score(ytest,y_pred_sk)
    print("sklearn's r2",r2_sk)

    #Plot
    plt.scatter(xtrain,ytrain,facecolor="blue",marker="o",label="Train set",alpha=0.7)
    plt.scatter(xtest,ytest,facecolor="red",marker="o",label="Test set",alpha=0.5)
    #Smooth model curve
    x_trans = np.transpose(xtest)
    y_pred_trans = np.transpose(y_pred)
    c1=interp1d(x_trans[0],y_pred_trans[0],kind="cubic")
    xtest_smooth=np.linspace(x_trans[0].min(),x_trans[0].max(),100)
    ytest_smooth=c1(xtest_smooth)
    #Smooth sklearn curve
    c2=interp1d(x_trans[0],np.transpose(y_pred_sk)[0],kind="cubic")
    y_sk_smooth=c2(xtest_smooth)

    plt.plot(xtest_smooth,ytest_smooth,color="green",linewidth=2,label="Model prediction")
    #plt.plot(xtest_smooth,y_sk_smooth,color="orange",linestyle="--",label="Sklearn prediction")
    plt.xlim([-4,4])
    plt.ylim([-4,10])
    plt.title(f"Gradient descent polynomial regression lr {gd.lr}, epochs {gd.epochs}")
    plt.legend()

    plt.show()