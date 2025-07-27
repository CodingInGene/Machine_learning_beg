#Binary classification algorithm
from sklearn.datasets import make_classification as mclassi
from matplotlib import pyplot as plt
import numpy as np

#Function block
def traintestsplit(x,y,splitby=0.5):
    n=len(x)
    return x[:round(n*splitby)],y[:round(n*splitby)],x[round(n*splitby):],y[round(n*splitby):]  #xtrain,ytrain,xtest,ytest

def step(z):
    #return 1 if z>0 else 0
    return (1/(1+np.exp(-z)))    #Sigmoid

class Perceptron:
    def __init__(self):
        self.weights = None
        self.epochs = None
        self.lr = None
        self.y_pred = None

    def fit(self,x,y,learning_rate,epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.ones([x.shape[1]])    #Contains intercept, coeffs (w0,w1,w2)

        for i in range(0,epochs):
            idx = np.random.randint(0,x.shape[0])

            y_hat = step( np.dot(x[idx], self.weights) )
            self.weights += self.lr * ((y[idx]-y_hat) * x[idx])
            #print(y[idx]-y_hat)

    def predict(self,xtest):
        self.y_pred = []
        for i in np.dot([self.weights],np.transpose(xtest))[0]:
            if i>=0:
                self.y_pred.append(1)
            else:
                self.y_pred.append(0)

    # def logTolinTransformer(self):
    #     m = -(self.weights[2]/self.weights[1])
    #     b = -(self.weights[0]/self.weights[1])
    #     return m,b
            


#Main block
if __name__ == "__main__":
    X,Y = mclassi(n_samples=100,n_features=2,n_informative=1,n_redundant=0,n_classes=2,n_clusters_per_class=1,class_sep=20,hypercube=False,random_state=41)
    # X-> x1,x2 Nx2, Y-> Nx1

    #Test train split
    xtrain,ytrain,xtest,ytest = traintestsplit(X,Y,splitby=0.8)
    print(xtrain.shape,xtest.shape)

    #Add bias term in xtrain, xtest
    x_train_new=np.transpose([ np.ones(xtrain.shape[0]),np.transpose(xtrain)[0],np.transpose(xtrain)[1] ])    # Nx3, x0,x1,x2
    x_test_new=np.transpose([ np.ones(xtest.shape[0]),np.transpose(xtest)[0],np.transpose(xtest)[1] ])

    #Predict
    lreg = Perceptron()
    lreg.fit(x_train_new,ytrain,learning_rate=0.1,epochs=1000)
    lreg.predict(x_test_new)
    print("Weights",lreg.weights)

    #Transform weights to linear equation
    # m=B/C, b=A/C
    m = -(lreg.weights[1]/lreg.weights[2])
    b = -(lreg.weights[0]/lreg.weights[2])

    x_line = np.linspace(-3,3,xtest.shape[0])
    y_line = (m * x_line) + b

    #Wrong points
    print(ytest - lreg.y_pred)
    wrong = ytest - lreg.y_pred
    wcount=0
    for i in wrong:
        if i != 0:
            wcount+=1

    #Plot
    plt.scatter(np.transpose(xtrain)[0],np.transpose(xtrain)[1],cmap="winter",c=ytrain,s=50)
    plt.plot(x_line,y_line,color="red",linewidth=2)
    plt.ylim([5,-5])
    plt.title(f"Logistic regression with perceptron trick\nEpochs {lreg.epochs}, learning rate {lreg.lr}, Errors {wcount}")
    plt.show()