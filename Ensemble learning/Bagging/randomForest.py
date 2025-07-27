#Bagging using Decision trees
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from collections import defaultdict
import time

#Function block
def traintestsplit(x,y,trainrows):
    return x[:round(x.shape[0]*trainrows)],y[:round(x.shape[0]*trainrows)],x[round(x.shape[0]*trainrows):],y[round(x.shape[0]*trainrows):]

class Ensemble:
    def __init__(self):
        self.max_samples = None     #Defines how much train set size to assign
        self.sets = None    #Defines how many classifiers to create
        self.models = None

    def fit_models(self,max_samples,n_estimators,xtrain,ytrain):
        self.max_samples = max_samples  #How much of train data to feed every classifier
        self.n_estimators = n_estimators

        self.models = [] #Array containing all classifiers object

        #Create classifiers
        for i in range(0,self.n_estimators):
            decision = DecisionTreeClassifier()
            self.models.append(decision)

        #Fit classifiers
        for i in range(0,self.n_estimators):
            samples = np.random.choice(xtrain.shape[0],self.max_samples,replace=True) #[np.random.randint(0,xtrain.shape[0]) for i in range(0,self.max_samples)] #Randomly select training data
            self.models[i].fit(xtrain[samples],ytrain[samples])

    def predict(self,xtest):
        #Predictions
        label = LabelEncoder()  #To label encode y_pred values

        pred = []  #Contains all prediction matrices of all classifiers
        for i in range(0,self.n_estimators):
            pred.append(label.fit_transform(self.models[i].predict(xtest)))     # (N x No. of classifiers) matrix
        
        #Voting
        pred = np.transpose(pred)
        print(pred)

        self.y_pred = []
        for i in pred:
            d=defaultdict(int)
            for j in i: #Count elements frequency
                d[j] += 1
            max_v = 0   #Find max voted prediction
            for j in d.keys():
                if d[j]>max_v:
                    max_v = d[j]
                    max_k = j
            self.y_pred.append(max_k)
        self.y_pred = np.array(self.y_pred)


#Main block
if __name__=="__main__":
    mnist = fetch_openml(name="mnist_784",version=1)
    X = mnist["data"].to_numpy()
    Y = mnist["target"].to_numpy()

    label = LabelEncoder()
    Y_new = label.fit_transform(Y)
    
    #Train test split
    xtrain,ytrain,xtest,ytest = traintestsplit(X,Y_new,trainrows=0.8)

    time_init = time.time()

    #Ensemble training
    ens = Ensemble()
    ens.fit_models(max_samples=10,n_estimators=2,xtrain=xtrain,ytrain=ytrain)

    #Prediction
    ens.predict(xtest)
    print(ens.y_pred)

    #Accuracy
    acc = accuracy_score(ytest,ens.y_pred)
    print("Ensemble models Accuracy score",round(acc,4))

    #time taken
    time_end = time.time()
    print(f"Time taken {round(time_end-time_init,3)}s")

    #Sklearn
    des = DecisionTreeClassifier()
    des.fit(xtrain,ytrain)
    y_sk = des.predict(xtest)
    acc_sk = accuracy_score(ytest,y_sk)
    print("\nSklearn Decision tree accuracy score",round(acc_sk,4))