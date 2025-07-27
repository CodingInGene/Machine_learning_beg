#Softmax regression
import numpy as np
from sklearn.datasets import load_iris,fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as ttsp
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib import pyplot as plt
import time
from collections import defaultdict

#Function block
def loadData(dataset):
    if dataset=="iris":
        iris=load_iris()
        x=np.transpose([iris["data"][:,0],iris["data"][:,2]])    #sepal and petal length, (Nx(m+1))
        y=iris["target"]

    elif dataset=="mnist":
        mnist=fetch_openml("mnist_784",version=1)
        x=mnist["data"]
        y=np.array(list(map(int,mnist["target"])))

    return x,np.transpose([y])

def traintestsplit(x,y,trainrows):
    #return [x[np.random.randint(0,x.shape[0])] for i in range(0,trainrows)],y[:trainrows],x[trainrows:],y[trainrows:]
    # np.random.shuffle(x)
    # np.random.shuffle(y)
    return x[:round(x.shape[0]*trainrows)],y[:round(x.shape[0]*trainrows)],x[round(x.shape[0]*trainrows):],y[round(x.shape[0]*trainrows):]

def softmaxfunc(z):
    if len(z.shape) > 1:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    else:   #For single image
        exp_z = np.exp(z - z.max())
        return exp_z / np.sum(exp_z)

class Softmax:
    def __init__(self):
        self.y_encoded=None
        self.y_label_encoded=None
        self.labels=None
        self.epochs=None
        self.lr=None
        self.weights=None
        self.kroneckerDelta=None   #Not used
        self.slope=None
        self.y_pred=None
        self.all_prob=None
    def hotencoder(self,y):    #One hot encoding, takes Nx1 matrix
        #Finding unique classes
        no_of_classes = len(np.unique(y))
        self.y_encoded=np.zeros([y.shape[0],no_of_classes])

        #Assign labels to each class
        n=np.unique(y)
        d=defaultdict(int)
        count=0
        for i in np.unique(y):
            d[i] = count
            count+=1

        #Encode
        for i in range(0,self.y_encoded.shape[0]):
            label = d[ y[i][0] ]
            self.y_encoded[i][label] = 1
    def labelencoder(self,y):
        self.y_label_encoded = []

        #Assign labels to each class
        n=np.unique(y)
        d=defaultdict(int)
        count=0
        for i in np.unique(y):
            d[i] = count
            count+=1

        for i in range(0,y.shape[0]):
            self.y_label_encoded.append( d[y[i][0]] )
        
        self.labels = d     #Categorical to labels

        self.y_label_encoded = np.transpose( [np.array(self.y_label_encoded)] )
    def labeldecoder(self,y):
        decoded = []
        for i in y:
            for key,val in self.labels.items():
                if val == i:
                    decoded.append(key)
        return decoded
    def fit(self,x,learning_rate,epochs,batch_size):
        self.epochs=epochs
        self.lr=learning_rate
        self.weights=np.zeros([y_encoded.shape[1],x.shape[1]])  # no of classes x no of features+1

        #Kronecker delta function
        '''
        Making Kronecker delta matrix by shape of X dot Weights -> Nx(m+1) (where m = no of classes), as it's dim are same as y_hat
        Filling diagonals with 1, non diagonals with 0
        '''
        # self.kroneckerDelta = np.zeros(y_encoded.shape)
        # for i in range(0,x.shape[0]):
        #     for j in range(0,x.shape[1]):
        #         if i==j:
        #             self.kroneckerDelta[i][j]=1
        
        for i in range(0,self.epochs):
            batch=np.array( [np.random.randint(0,x.shape[0]) for i in range(batch_size)] )
            #print(x[batch])

            y_hat = softmaxfunc(np.dot(x[batch],np.transpose(self.weights)))
            '''
            self.slope = np.dot(np.dot( np.transpose(self.y_encoded), np.transpose(np.transpose(self.kroneckerDelta) - y_hat) ), x)
            Formula -> slope=Y.[Kronecker - Y_hat].X
            Not using kronecker because we are using vector form, kronecker is used in derivation of loss func.
            Adding kronecker here causes output to Nx(m+1)
            Now formula is (Y-y_hat)*X
            '''
            self.slope = np.dot(np.transpose(y_encoded[batch] - y_hat),x[batch])
            self.weights += self.lr * self.slope

    def predict(self,xtest):    #Takes Nx(m+1)
        self.all_prob = softmaxfunc(np.dot(xtest,np.transpose(self.weights)))    #Probability of all classes
        if len(self.all_prob.shape) > 1:
            self.y_pred = np.argmax(self.all_prob,axis=1)   #Get which col(class) has max probability
        else:   #For single image
            self.y_pred = np.argmax(self.all_prob)


#Main block
if __name__=="__main__":
    start_time=time.time()

    dataset="mnist"
    X,Y=loadData(dataset=dataset)

    softmax=Softmax()

    #Train test split
    xtrain,ytrain,xtest,ytest = traintestsplit(X,Y,trainrows=0.8)


    #Change X for bias term
    xtrain_new = np.insert(xtrain,obj=0,values=1,axis=1)
    xtest_new = np.insert(xtest,obj=0,values=1,axis=1)
    print(xtrain_new.shape,ytest.shape)

    #One hot encoding
    softmax.hotencoder(ytrain)
    y_encoded=softmax.y_encoded
    #print(y_encoded)

    #Training
    softmax.fit(xtrain_new,learning_rate=0.05,epochs=9000,batch_size=30)
    #print("Models prediction weight\n",softmax.weights)

    #Prediction
    softmax.predict(xtest_new)
    #print("Models y predicted",softmax.y_pred)

    '''
    Sklearn
    '''
    #Sklearns logistic regression
    logit = LogisticRegression()
    logit.fit(xtrain,np.transpose(ytrain)[0])
    #print("\nSklearns weights","Intercept",logit.intercept_,"Coef\n",logit.coef_)
    sk_pred = logit.predict(xtest)
    #print("Sklearns y predicted",sk_pred)

    #Error metrics
    model_conf = confusion_matrix(ytest,softmax.y_pred)
    acc_score=accuracy_score(ytest,softmax.y_pred)
    print("Models confusion matrix\n",pd.DataFrame(model_conf))

    sk_conf = confusion_matrix(ytest,sk_pred)
    sk_acc=accuracy_score(ytest,sk_pred)
    print("Sklearns confusion matrix\n",pd.DataFrame(sk_conf))

    #print("Labels",softmax.labels)  #Print encoded labels
    
    print("Models accuracy score",acc_score)
    print("Sklearns accuracy score",sk_acc)

    end_time=time.time()
    print("Time taken",end_time-start_time,"s")

    #Plot
    '''
    X.values[0] -> Get single image. X is pd dataframe/dict so extracting values from it then slicing
    '''
    #Individual image prediction
    idx=19
    softmax.predict(xtest_new[idx])
    print(f"Prediction of X[{idx}]",softmax.y_pred,",Actual",ytest[idx])
    print(f"Probability of prediction {softmax.all_prob}, {np.max(softmax.all_prob)*100}%")  #Probability
    plt.imshow(xtest_new[idx][1:].reshape(28,28), cmap="gray")
    plt.title(f"Idx {idx}\nPredicted {softmax.y_pred}")
    # plt.title("MNIST data")   #Plotting multiple images
    # for i in range(10):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(X.values[i].reshape(28,28), cmap="gray")

    # plt.imshow(model_conf,cmap="binary")
    # plt.title("Confusion matrix")
       
    plt.show()


'''
Note -
Use of important variables are mentioned in Softmax_notes (Main notes)
'''