import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
from matplotlib.image import imread
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans as SkKmeans
from sklearn.datasets import make_blobs
import math
import random
import time
from scipy.spatial.distance import cdist
import copy

#Function block
def loadData():
    img = imread("rubiks.jpeg")
    return img

class Kmeans:
    def __init__(self):
        self.dist=None  #Distance matrix
        self.clusters=None  #Cluster matrix
        self.inertia=None
        self.centroids=None
        self.ymeans=None    #Point's Cluster id

    def fit_predict(self,x,max_itr: int,n_clusters: int):    #Takes X, max iteration to perform in for loop and number of clusters to form
        '''Centroids are random points from x'''
        random_idx = random.sample(range(0,x.shape[0]),n_clusters)    #Generate random samples (x rows)
        self.centroids = x[random_idx]

        #prev_centroids = copy.deepcopy(self.centroids) #For early stopping

        '''Ask centroid calculations to give proper and final centroid'''
        for i in range(max_itr):
            #Calculate euclidian dist from every centroid to all points
            '''dist->[[c1d1,c1d2,],[c2d1,c2d2,]]'''
            self.dist = cdist(self.centroids, x, metric="euclidean")

            '''Assigning clusters (which clusters distance is least)'''
            self.clusters= [[] for i in range(0,n_clusters) ]  # [ [c1] [c2] [c3] ] -> Contains points, initially blank

            self.ymeans=np.argmin(np.transpose(self.dist),axis=1)   #Converting ClustersxN(2xN) to NxClusters(Nx2) to get min values
            #Assigning to clusters from argmin
            for i in range(0,len(self.ymeans)):
                '''
                indexes in ymeans denotes the point's cluster id. also ymeans index corresponds to X row index.
                *Each cluster array can have different sizes. So it must be kept as normal arr not numpy arr.
                Clusters -> [ [c1-> p1,p2,p3,p4],[c2-> p1,p2], ... ]
                '''
                self.clusters[self.ymeans[i]].append( x[i] )
            #self.clusters = x[self.ymeans]
            #Moving clusters
            '''
            Move clusters by calculating mean of x points (mean will be new centroids x), mean of y (y of new centroid).
            For each centroid
            '''
            print(self.centroids)
            for i in range(0,len(self.clusters)):
                if len(self.clusters[i]) == 0:
                    self.centroids[i][0]=0          #If cluster is empty then centroid 0
                    self.centroids[i][1]=0
                else:
                    new_coords=np.transpose(self.clusters[i])
                    print(new_coords)
                    self.centroids[i][0]=np.mean(new_coords[0])
                    self.centroids[i][1]=np.mean(new_coords[1])
            # print(prev_centroids,"Recent",self.centroids)     #Early stopping
            # if np.allclose(prev_centroids,self.centroids,rtol=1e-2):
            #     print("Hello",i)
            #     break
            # prev_centroids = copy.deepcopy(self.centroids)
        
        '''Calculating inertia or WCSS'''
        #Total sum of euclidian distances between each centroid and respective each clusters points
        self.inertia=0
        for i in range(0,self.centroids.shape[0]):
            self.inertia += np.sum(cdist([self.centroids[i]],self.clusters[i],metric="euclidean"))
            '''cdist only takes 2d arr. So [centroids[i]] makes it 2d -> [[x1,y1]]'''


#Main block
if __name__=="__main__":
    X=loadData()
    print(X.shape)

    X_new = X.reshape(-1,3)
    print(X_new.shape)
    #print(X_new)

    time_init = time.time()

    #Train
    kmeans = Kmeans()

    #Elbow curve
    # elbow=[]
    # for i in range(1,15):
    #     kmeans.fit_predict(X_new,max_itr=40,n_clusters=i)  # on 100 itr centroid values were nearly same on every exec
    #     elbow.append([i,kmeans.inertia])
    # elbow=np.array(elbow)

    # plt.plot(elbow[:,0],elbow[:,1])

    # #Predict
    no_of_clusters=6
    kmeans.fit_predict(X_new,max_itr=40,n_clusters=no_of_clusters)
    print("Centroid",kmeans.centroids)
    print("Clusters",kmeans.ymeans)

    kmeans_seg = kmeans.centroids[kmeans.ymeans]
    kmeans_seg = kmeans_seg.reshape(X.shape)
    clusts = X_new[kmeans.ymeans]
    print("New clusts",clusts)
    print("\nClusts",kmeans.clusters)

    time_end = time.time()
    print(f"Time taken {round(time_end-time_init,2)}s")


    #Sklearn
    time_init = time.time()

    sk = SkKmeans(n_clusters=no_of_clusters).fit(X_new)
    sk_seg = sk.cluster_centers_[sk.labels_]
    sk_seg = sk_seg.reshape(X.shape)
    print("\nSklearn centroid\n",sk.cluster_centers_)
    print(sk.labels_)

    time_end = time.time()
    print(f"Sklearn Time taken {round(time_end-time_init,2)}s\n")
    
    #Plot
    # segmented_img = []
    # for i in range(no_of_clusters):
    #     plt.imshow( np.array([X_new[kmeans.ymeans == i]]).reshape(X.shape) )   #Shape (1, 2185, 3)

    # print(segmented_img)
    # #segmented_img = np.array(segmented_img)
    # #print(segmented_img[0].reshape(X.shape))
    # print(segmented_img[0])
    # print(segmented_img[0].shape)
    # plt.imshow(segmented_img[0].reshape(X.shape))

    fig,ax = plt.subplots(nrows=1,ncols=3)

    ax[0].imshow(X)
    ax[1].imshow((sk_seg*255).astype(np.uint8),cmap="gray")
    ax[2].imshow((kmeans_seg*255).astype(np.uint8),cmap="gray")

    ax[0].set_title("Original")
    ax[1].set_title("Segmented using Sklearn")
    ax[2].set_title("Segmented using Model")
    fig.suptitle(f"Image clusterization\n\nNo. of clusters {no_of_clusters}")
    
    plt.show()