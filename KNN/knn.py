#Make KNN classifier
from sklearn.datasets import fetch_openml,make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.metrics import accuracy_score,confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mode
import time
from threading import Thread
from concurrent.futures import ProcessPoolExecutor
import os

#Function block
def loadData():
    mnist = fetch_openml(name="mnist_784",version=1)
    x = np.array(mnist["data"])
    y = np.array(mnist["target"])
    return x[:30000],y[:30000]
    # x,y=make_classification(n_samples=40000,n_features=10,n_informative=2,n_redundant=0,n_classes=3,n_clusters_per_class=1,class_sep=15,hypercube=False,random_state=41)
    # return x,y

def traintestsplit(x,y,train_size=0.5):
    n = train_size
    xtrain = x[:round(x.shape[0]*n)]
    xtest = x[round(x.shape[0]*n):]
    ytrain = y[:round(x.shape[0]*n)]
    ytest = y[round(x.shape[0]*n):]

    return xtrain,xtest,ytrain,ytest

def getNargmin(d,n):    # Not using now
    comb = zip(np.arange(len(d)),d)
    min_values = np.array(sorted(comb, key=lambda t:t[1]))[:n]  #Sorted returns list of tuples, np.array makes it list of lists
    return min_values[:,0].astype(int)  #Return only the indexes. ndarray astype(int) -> indexes should be int

'''Multiprocessing'''
def mprocessDist(xtest,xtrain,distance_metric="cityblock"):
    dist = cdist(xtest,xtrain,metric=distance_metric)
    return np.array(dist)

'''Multithreading'''
def threadPred(dist,k,ytrain):
    min_indices = np.argpartition(dist, k, axis=1)[:, :k]     #Nearest neighbors upto k
    neighbor_labels = ytrain[min_indices]  # Get labels of nearest neighbors
    y_pred = mode(neighbor_labels, axis=1, keepdims=False).mode    # Vectorized majority vote

    return np.array(y_pred)

class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None, args={}, kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._value = None  #Return value var
    def run(self):
        if self._target != None:
            self._value = self._target(*self._args, **self._kwargs)
    def join(self):
        Thread.join(self)
        return self._value
    '''Default join,run methods doesnot return anything, so we made custom thread modules for return values'''

class KNearestNeighbour:
    def __init__(self, n_neighbours, n_jobs=3, metric="cityblock"):
        self.k = n_neighbours
        self.y_pred = None
        self.xtrain = None
        self.ytrain = None
        self.n_jobs = n_jobs
        self.distance_metric = metric
    def fit(self,xtrain,ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain
    def predict(self,xtest):
        self.y_pred=[]
        if len(xtest) > 1:
            time_init = time.time()
            #dist = cdist(xtest,self.xtrain,metric=self.distance_metric)  # a=xtest, b=xtrain
            '''Threading for dist will increase time. Don't use'''

            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                xtest_div = np.split(xtest, self.n_jobs)
                xtrain_div = np.split(self.xtrain, self.n_jobs)
                results = executor.map(mprocessDist, np.split(np.tile(xtest,(self.n_jobs,1)), self.n_jobs), xtrain_div)     #results is a generator iterable

            dist = np.hstack(list(results))   #converting generator object to list

            time_end = time.time()
            print(f"Time taken for dist calcu {round(time_end-time_init,2)}s")

            # for distance in dist:
            #     min_indices = getNargmin(distance,self.k)  #Get indexes of n min values

            #     values, count = np.unique(self.ytrain[min_indices], return_counts=True)
            #     majority = values[np.argmax(count)]    #Get which class repeats most in neighbour
            #     self.y_pred.append(majority)
            time_init = time.time()
            
            '''Without threading'''
            # min_indices = np.argpartition(dist, self.k, axis=1)[:, :self.k]     #Nearest neighbors upto k
            # neighbor_labels = self.ytrain[min_indices]  # Get labels of nearest neighbors
            # self.y_pred = mode(neighbor_labels, axis=1, keepdims=False).mode    # Vectorized majority vote

            '''Threading for y_pred'''
            '''With threading'''
            div = np.split(dist,4)
            t1 = CustomThread(target=threadPred,args=(div[0], self.k, self.ytrain))
            t2 = CustomThread(target=threadPred,args=(div[1], self.k, self.ytrain))
            t3 = CustomThread(target=threadPred,args=(div[2], self.k, self.ytrain))
            t4 = CustomThread(target=threadPred,args=(div[3], self.k, self.ytrain))
            t1.start()
            t2.start()
            t3.start()
            t4.start()
            temp1 = t1.join()   #Get return values
            temp2 = t2.join()
            temp3 = t3.join()
            temp4 = t4.join()

            # with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            #     results = executor.map(threadPred, div, np.repeat(self.k, self.n_jobs), np.split(np.tile(self.ytrain,(self.n_jobs)),self.n_jobs))

            self.y_pred = np.concatenate((temp1,temp2,temp3,temp4), axis=0)
            #self.y_pred = np.array(list(results)).reshape(-1)   #Flattening returned values

            time_end = time.time()
            print(f"Time taken for y_pred {round(time_end-time_init,2)}s")
        else:
            dist = cdist(xtest,self.xtrain,metric=self.distance_metric)  # a=xtest, b=xtrain
            min_indices = np.argpartition(dist, self.k, axis=1)[:, :self.k]     #Nearest neighbors upto k
            neighbor_labels = self.ytrain[min_indices]  # Get labels of nearest neighbors
            self.y_pred = mode(neighbor_labels, axis=1, keepdims=False).mode    # Vectorized majority vote
            


#Main block
if __name__=="__main__":
    X,Y = loadData()
    Y_new = LabelEncoder().fit_transform(Y) #Label encoded categorical data

    #Train test split
    xtrain,xtest,ytrain,ytest = traintestsplit(X,Y_new,train_size=0.7)
    print(xtrain.shape,ytest.shape)

    time_init = time.time()

    #KNN
    print("Available cpus",os.cpu_count())
    knn = KNearestNeighbour(n_neighbours=3,n_jobs=5,metric="cityblock")
    knn.fit(xtrain,ytrain)
    knn.predict(xtest)

    time_end = time.time()

    #Confusion matrix
    conf = pd.DataFrame(confusion_matrix(ytest,knn.y_pred))
    print("Models confusion matrix\n",conf)
    #Accuracy score
    acc = accuracy_score(ytest, knn.y_pred)
    print(f"Model accuracy score {round(acc,4)}")
    print(f"Time taken {round(time_end-time_init,2)}s")

    
    #Sklearn
    time_init = time.time()

    sk_knn = KNeighborsClassifier(n_neighbors=3)
    sk_knn.fit(xtrain,ytrain)
    y_pred = sk_knn.predict(xtest)

    #Confusion matrix
    conf = pd.DataFrame(confusion_matrix(ytest,y_pred))
    print("\nSklearns confusion matrix\n",conf)
    #Accuracy score
    acc_sk = accuracy_score(ytest, y_pred)
    print(f"Sklearn accuracy score {round(acc_sk,4)}")

    time_end = time.time()
    print(f"Time taken {round(time_end-time_init,2)}s")

    #Single image prediction
    idx = 241

    knn.predict(xtest[idx].reshape(1,-1))

    plt.imshow(xtest[idx].reshape(28,28), cmap="gray")
    plt.title(f"Idx {idx}\nPredicted {knn.y_pred[0]}, Actual {ytest[idx]}\nModel's accuracy {round(acc,4)*100}%")
    plt.show()