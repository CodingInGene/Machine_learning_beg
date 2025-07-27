# K Nearest Neighbour

**Variables**

self.n_jobs - for multiprocessing in distance calculations (cdist)


**cdist -**

```
x=[[xp1],[xp2],[xp3]]   _Known train x points_
y=[[p1],[p2]]   _New points(test)_

Output -
[
    [p1 xp1     p1 xp2      p1 xp3]     _Distances with p1 and xtrain points_
    [p2 xp1     p2 xp2      p2 xp3]     _Distances with p2 and xtrain points_
]
```


**Program**

```
    def getNargmin(d,n):
        comb = zip(np.arange(len(d)),d)
        min_values = np.array(sorted(comb, key=lambda t:t[1]))[:n]  #Sorted returns list of tuples, np.array makes it list of lists
        return min_values[:,0].astype(int)  #Return only the indexes. ndarray astype(int) -> indexes should be int

    def predict(self,xtest):
        self.y_pred=[]
        dist = cdist(xtest,self.xtrain,metric="minkowski")  #Get euclidean distance between new point and known points
        for distance in dist:
            min_indices = getNargmin(distance,self.k)  #Get indexes of n min values

            values, count = np.unique(self.ytrain[min_indices], return_counts=True)
            majority = values[np.argmax(count)]    #Get which class repeats most in neighbour
            self.y_pred.append(majority)
        self.y_pred = np.array(self.y_pred)
```


**v1**

* In predict function-
    dist - calculates distances between new (xtest) points and all xtrain points
    min_indices holds values of k no of nearest neighbours (list).
    np.unique returns all unique numbers and their occurance
    majority - get max values(max in count) index and get respective value from ytrain, to get majority count in ytrain

* In getNargmin-
    1. comb - zip 0-len(d) and distances together (indices and distances,eg 0,1,2,3 and 1.2, 45, 9.2, 10.1)
    2. _d_ -> [64.19501538  3.          3.          3.        ]

    3. min_values = np.array(sorted(comb,key=lambda t:t[1])) -> 
       Sorts by distance (2nd column)
        [ 1.          3.        ]   **left side - indices, right side - distances**
        [ 2.          3.        ]
        [ 3.          3.        ]
        [ 0.         64.19501538]

    4. Return only index column


**v2, vectorization**

```
    def predict(self,xtest):
        min_indices = np.argpartition(dist, self.k, axis=1)[:, :self.k]
        
        neighbor_labels = self.ytrain[min_indices]  # Get labels of nearest neighbors
        
        self.y_pred = mode(neighbor_labels, axis=1, keepdims=False).mode    # Vectorized majority vote
```

**v3, Multithreading in nearest neighbors**
_Threads 4_

```
    def threadPred(dist,k,ytrain):
        min_indices = np.argpartition(dist, k, axis=1)[:, :k]     #Nearest neighbors upto k
        neighbor_labels = ytrain[min_indices]  # Get labels of nearest neighbors
        y_pred = mode(neighbor_labels, axis=1, keepdims=False).mode    # Vectorized majority vote

        return np.array(y_pred)

    KNN class, predict method
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

    self.y_pred = np.concatenate((temp1,temp2,temp3,temp4), axis=0)
```


**v4, Multi-core processing in distance calculation**

```
    def mprocessDist(xtest,xtrain,distance_metric="cityblock"):
        dist = cdist(xtest,xtrain,metric=distance_metric)
        return np.array(dist)

    Predict method
    with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        xtest_div = np.split(xtest, self.n_jobs)
        xtrain_div = np.split(self.xtrain, self.n_jobs)
        results = executor.map(mprocessDist, np.split(np.tile(xtest,(self.n_jobs,1)), self.n_jobs), xtrain_div)     #results is a generator iterable

    dist = np.hstack(list(results))   #converting generator object to list
```



**Notes**

1. Calculation time increases with increasing _features_ and _no of rows_.
    With 10,000 rows, 5 features,
        Model accuracy score 0.97
        Time taken 0.34s
        Sklearn accuracy score 0.97
        Time taken 0.1s
    With 10,000 rows, 10 features,
        Model accuracy score 0.97
        Time taken 0.38s
        Sklearn accuracy score 0.97
        Time taken 0.21s
    With 20,000 rows, 5 features,
        Model accuracy score 0.99
        Time taken 1.12s
        Sklearn accuracy score 0.99
        Time taken 0.2s
    With 30,000 rows, 10 features,
        Model accuracy score 1.0
        Time taken 2.96s
        Sklearn accuracy score 1.0
        Time taken 0.95s

2. In mnist, row 10,000
    Model accuracy score 0.93
    Time taken _27.73s_
    Sklearn accuracy score 0.93
    Time taken 0.89s

3. For mnist, best accuracy can be found with n_neighbours = 3

4. For KNearestNeighbour class, distance metrics - "euclidean", "minkowski", "cityblock"
    Default is "cityblock" (Manhattan distance)



**Time**

With 40,000 rows, 10 features, (Without threading)
    Model took 4.45s
    Sklearn 1.56s

With 40,000 rows, 10 features, (With threading, 4 threads) **v3**
    Model took 2.75s
    Sklearn 1.56s

With 30,000 rows (21,000 train set), MNIST **v4**
    Time taken for dist calcu 55.48s
    Time taken for y_pred 0.84s
    Model accuracy score 0.9488
    Time taken 56.33s



**v3**

In mnist, row 10,000 (7000 train set), all features(782)
    Time taken for dist calcu 23.78s (1500, 7000)   **Distance calculations are taking most time**
    Time taken for y_pred 0.19s
    Time taken by model 23.97s

    but sklearn took only 0.93s


**v4**

cores - 5   _Seems best for mnist_
Time taken for dist calcu 8.16s     **Reduced due to multiprocessing**
Time taken for y_pred 0.18s
Model accuracy score 0.92
Time taken by model 8.34s

Sklearn accuracy score 0.93
Sklearn Time taken 0.94s
