# Dataset

# Iris
**total 150**

**learning_rate=0.05,epochs=8500**  (optimum)

**v1**
test set 30
Models confusion matrix
    0   1
0  0   0
1  7  23
Sklearns confusion matrix
    0   1
0  0   0
1  6  24

**v2**
test set 75
**test 1**
Models confusion matrix
     0   1  2
0  12  12  0
1  10  18  0
2  12  11  0
Sklearns confusion matrix
     0  1   2
0   9  3  12
1   8  6  14
2  11  1  11

**test 2**
Models confusion matrix
     0  1  2
0  16  0  7
1  20  0  7
2  21  0  4
Sklearns confusion matrix
     0  1  2
0  14  1  8
1  18  0  9
2  19  0  6

**test 3**
Models confusion matrix
    0   1   2
0  0  10  17
1  0  11  15
2  0   8  14
Sklearns confusion matrix
    0   1   2
0  0  11  16
1  0  11  15
2  0   8  14


**Problem due to mlxtend plotting**
test set 30
Sklearns confusion matrix
    0   1
0  0   0
1  9  21
It changed after implementing mlxtend's plotting descision regions. Models conf matrix was also became same as sklearns but it changed back to original when tweaked the lr and epochs.
Prev it was -
    0   1
0  0   0
1  6  24

**Warning**
UserWarning: A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.

Reason- test set of len 30 had only class 2. And model predicted it all correctly so confusion matrix had only one label.
epochs 80, lr 0.05, batch size 30->
Models prediction weight
 [[ 0.44829446  1.2746577  -2.028784  ]
 [-0.05633705 -0.32310419  0.3356814 ]
 [-0.39195742 -0.95155351  1.6931026 ]]
Models y predicted [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
Models confusion matrix
     0
0  30

Solution- shuffling

After shuffling-
learning_rate=0.05,epochs=200,batch_size=30->
Models confusion matrix
    0   1  2
0  0  13  0
1  0  11  0
2  0   6  0
Sklearns confusion matrix
    0  1  2
0  7  3  3
1  3  4  4
2  1  3  2