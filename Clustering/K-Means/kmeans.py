import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import math
import random
from scipy.spatial.distance import cdist

#Function block
def loadData(dataset):
    if dataset=="cgpa_vs_iq":
        stud=pd.read_csv("cgpa_vs_iq.csv")
        x=np.transpose([stud["CGPA"],stud["IQ"]])
        return x
    if dataset=="blob":
        cstd=[1,1,1,1]
        centroids=[(-5,-5),(5,5),(-1,-1),(1,1)]
        x,y=make_blobs(n_samples=100,cluster_std=cstd,centers=centroids,n_features=2,random_state=2)
        return x

class Kmeans:
    def __init__(self):
        self.dist=None  #Distance matrix
        self.clusters=None  #Cluster matrix
        self.inertia=None
        self.centroids=None
        self.ymeans=None    #Point's Cluster id
    def euclidianDist(self,points: [[int,int],[int,int]]) -> float :    #takes n dim array points
        sum=0
        for i in np.transpose(points):
            sum+=pow(np.diff(i),2)    # a=[x1,x2]. np.diff(a) => x2-x1.
        d=math.sqrt(sum[0])
        return d

    def fit_predict(self,x,max_itr: int,clusters: int):    #Takes X, max iteration to perform in for loop and number of clusters to form
        '''Centroids are random points from x'''
        random_idx = random.sample(range(0,x.shape[0]),clusters)    #Generate random samples (x rows)
        self.centroids = x[random_idx]    #np.array([[1,i] for i in range(1,clusters+1) ]) , np.array([[0,0],[1,1],[2,2]]). np.zeros([clusters,2]) -prev

        '''Ask centroid calculations to give proper and final centroid'''
        for i in range(max_itr):  #Run it multiple times to check for centroid changes

            '''Distance from each centroid'''
            dt=[]  # dist->[[c1d1,c1d2,],[c2d1,c2d2,]]  Has rows = no of clusters, c1 distances will be put in dist[0]
            
            #Calculate euclidean dist from every centroid to all points
            self.dist = cdist(self.centroids, x, metric="euclidean")

            '''Assigning clusters (which clusters distance is least)'''
            self.clusters= [[] for i in range(0,clusters) ]  # [ [c1] [c2] [c3] ] -> Contains points, initially blank

            self.ymeans=np.argmin(np.transpose(self.dist),axis=1)   #Converting ClustersxN(2xN) to NxClusters(Nx2) to get min values loc
            #Assigning to clusters from argmin
            for i in range(0,len(self.ymeans)):
                '''
                indexes in ymeans denotes the point's cluster id. also ymeans index corresponds to X row index.
                *Each cluster array can have different sizes. So it must be kept as normal arr not numpy arr.
                Clusters -> [ [c1-> p1,p2,p3,p4],[c2-> p1,p2], ... ]
                '''
                self.clusters[self.ymeans[i]].append( x[i] )
            #Moving clusters
            '''
            Move clusters by calculating mean of x points (mean will be new centroids x), mean of y(y of new centroid). 
            For each centroid
            '''
            for i in range(0,len(self.clusters)):
                if len(self.clusters[i]) == 0:
                    self.centroids[i][0]=0          #If cluster is empty then centroid 0
                    self.centroids[i][1]=0
                else:
                    new_coords=np.transpose(self.clusters[i])
                    self.centroids[i][0]=np.mean(new_coords[0])
                    self.centroids[i][1]=np.mean(new_coords[1])
        
        '''Calculating inertia or WCSS'''
        #Total sum of euclidean distances between each centroid and respective each clusters points
        self.inertia=0
        for i in range(0,self.centroids.shape[0]):
            self.inertia += np.sum(cdist([self.centroids[i]],self.clusters[i],metric="euclidean"))
        '''cdist only takes 2d arr. So [centroids[i]] makes it 2d -> [[x1,y1]]'''


#Main block
if __name__=="__main__":
    X=loadData(dataset="blob")
    #print(X)

    #Train
    kmeans = Kmeans()

    #Elbow curve
    elbow=[]
    for i in range(1,15):
        kmeans.fit_predict(X,max_itr=40,clusters=i)  # on 100 itr centroid values were nearly same on every exec
        elbow.append([i,kmeans.inertia])
    elbow=np.array(elbow)

    #Predict
    no_of_clusters=3
    kmeans.fit_predict(X,max_itr=40,clusters=no_of_clusters)
    print("Centroids",kmeans.centroids)
    print("Clusters",kmeans.ymeans)
    
    #Plot
    fig = plt.figure()
    gs=gd.GridSpec(2,2, figure=fig)     #Spanning two cols for top plot
    ax1=fig.add_subplot(gs[0, :])
    ax2=fig.add_subplot(gs[1,0])
    ax3=fig.add_subplot(gs[1,1])

    ax1.plot(elbow[:,0],elbow[:,1])

    colors=["red","orange","blue","green","brown","yellow","pink","gray","cyan","purple"]
    for i in range(no_of_clusters):
        ax2.scatter(X[kmeans.ymeans==i,0],X[kmeans.ymeans==i,1],color=colors[i])
    ax3.scatter(X[:,0],X[:,1])

    ax1.set_title("Elbow curve")
    ax2.set_title(f"Clustering, No of clusters - {no_of_clusters}")
    ax3.set_title("X data")

    # ax2.set_xlabel("cgpa")
    # ax2.set_ylabel("IQ")
    # ax3.set_xlabel("cgpa")
    # ax3.set_ylabel("IQ")
    plt.show()

'''With 100 points took 18s to cluster'''