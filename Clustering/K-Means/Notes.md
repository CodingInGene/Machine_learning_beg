# Method variables

  **vars** - 
    X no of rows - N
    no of features - M

1. self.ymeans - Contains cluster id of each point in X. e.g - [0 1 0 1 2 ... 1 2 0 0 2]. _(ymeans = labels in sklearn)_
    (1xN)

2. self.clusters - Contains all points assigned to each clusters. e.g -
[ [ c1-> [p1(x,y)],[p2(x,y)]...[pn] ], [ c2-> [p1(x,y)],[p2(x,y)]...[pn] ], ... ]
[
  [array([4.34674973, 5.84245628]), array([5.69511961, 2.96653345]), ... array([5.43349633, 6.27837923])], **c1**
  [array([1.36505352, 1.77408203]), array([-0.20082, 0.26261366]), ... array([1.18040981, 1.55316427])], **c2**
  [array([-5.3135082, -4.22898826]), array([-4.49711858, -6.24528809]), ... array([-4.74342955, -5.98877905])] **c3**
  .
  .
  **cn**
]

3. self.centroids - ontains centroids - [ [centroid1 x,y,z...], [centroid2 x,y,z...], ...]
  self.centroids = x[random_idx] -> selects random rows from X.
  rows -> no of clusters
  cols -> no of features in X (M)
  (no of clusters x M)


# Time

1. With max_itr 25,

Clusters [4 4 4 ... 4 4 4]
Time taken 0.61s
Sklearn Time taken 0.09s

2. With max_itr 40,

Clusters [1 1 1 ... 3 3 3]
Time taken 1.0s
Sklearn Time taken 0.09s



**Notes**
1. Using for loop, sqrt to calculate euclidian distance increases time by 1000x(Sklearn is taking 0.09s, without elbow).
  nested loops-
    _Calling func in nested for loop, so O(n*m*2), for 40,000 - 40,000*3*2
    def euclidianDist(self,points: [[int,int],[int,int]]) -> float :    #takes n dim array points
        sum=0
        for i in np.transpose(points):
            sum+=pow(np.diff(i),2)    # a=[x1,x2]. np.diff(a) => x2-x1.
        d=math.sqrt(sum[0])
        return d

    _Used scipy cdist_ - It takes 2 2d arrays

2. After using cdist program became 200x fast than previous (without elbow). With nested loops time taken 402 seconds.

3. Changed max_itr to 40. Took 1/2x time than previous (100x slower than sklearn). Generally 40 is enough for more most cases.
  Changed max_itr to 25. 6x slower than sklearn.