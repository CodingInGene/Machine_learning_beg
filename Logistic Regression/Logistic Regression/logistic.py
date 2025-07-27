#Apply logistic regression
import numpy as np
from sklearn.datasets import load_iris, make_classification as mkclass
from matplotlib import pyplot as plt

#Function block
def loadData(dataset="iris",start=0,features=2,rows=100):
    if dataset=="iris":
        iris = load_iris()      #Dictionary
        y=[]
        for i in iris["target"][:rows]:
            if i==1:    #setosa
                y.append(1)
            else:
                y.append(0)
        return iris["data"][:rows][:,start :(start+features)], np.array(y)  # x :,0 -> starting col, :start+features -> ending column (:,0 :0+2)

    elif dataset=="sk":
        data=mkclass(n_samples=rows,n_features=features,n_informative=1,n_redundant=0,n_classes=2,n_clusters_per_class=1,class_sep=10,hypercube=False,random_state=41)
        return data[0],data[1]

def traintestsplit(x,y,splitby=0.5):
    return x[:round(x.shape[0]*splitby)],y[:round(x.shape[0]*splitby)],x[round(x.shape[0]*splitby):],y[round(x.shape[0]*splitby):]

def sigmoid(zarr):     #Takes array of z values, 1xN
    s=[]
    for i in zarr:
        sigmoidval = 1/(1+np.exp(-i))
        s.append(sigmoidval)
    return s

class LogisticReg:
    def __init__(self):
        self.lr=None
        self.epochs=None
        self.weights=None
        self.y_pred=None
    
    def fit(self,xtrain,ytrain,epochs,learning_rate):
        self.epochs=epochs
        self.lr=learning_rate
        self.weights=np.ones(xtrain.shape[1])
        for i in range(0,epochs):
            y_hat = sigmoid( np.transpose(np.dot(xtrain,self.weights)) )
            self.weights += (self.lr/xtrain.shape[0]) * (np.dot(ytrain-y_hat,xtrain))

    def predict(self,xtest):
        self.y_pred=[]
        for i in np.dot(xtest,self.weights):
            # if i>=0.5:
            #     self.y_pred.append(1)
            # else:
            #     self.y_pred.append(0)
            self.y_pred.append(i)
        return self.y_pred


#Main block
if __name__=="__main__":
    dataset="sk"
    X,Y = loadData(dataset=dataset,start=2,features=2,rows=200)

    #Split train test
    xtrain,ytrain,xtest,ytest = traintestsplit(X,Y,splitby=0.5)
    xtrain = np.insert(xtrain,obj=0,values=1,axis=1) #obj - where to insert, values - what to insert
    xtest = np.insert(xtest,obj=0,values=1,axis=1)
    print(xtrain.shape,ytest.shape)

    #Train
    lreg = LogisticReg()
    lreg.fit(xtrain,ytrain,learning_rate=0.01,epochs=8000)
    lreg.predict(xtest)
    print("Weights",lreg.weights)

    #Transform weights to m,b
    m = -(lreg.weights[1]/lreg.weights[2])
    b = -(lreg.weights[0]/lreg.weights[2])
    x_line = np.linspace(-3,3,xtest.shape[0])
    y_line = m*x_line + b

    #Plot
    if dataset=="iris":
        plt.scatter(xtrain[:,1],xtrain[:,2],c=ytrain,cmap="winter",s=xtrain.shape[0])
        #plt.scatter(xtrain[:,3],xtrain[:,4],c=ytrain,facecolor="grey")
        plt.xlim(0)
        plt.ylim(-1,2)
    else:
        plt.scatter(xtrain[:,1],xtrain[:,2],c=ytrain,cmap="winter",s=xtrain.shape[0])
        plt.ylim([-8,8])
    plt.plot(x_line,y_line,color="red",label="Regression line")
    plt.title(f"Logistic regression on {dataset} dataset\nLearning rate {lreg.lr} Epochs {lreg.epochs}")
    plt.legend()
    plt.show()