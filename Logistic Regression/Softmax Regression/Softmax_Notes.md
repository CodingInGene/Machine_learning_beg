This program uses mini batch GD algorithm to perform multinomial logistic regression(softmax).


**Train test split problem**

**v1**
split is not proper. function is splitting from start to a specific row. But classification data contains classes 0-2 in ordered manner. So where test dataset starts it may not have all classes data, then test set becomes improper.
Confusion matrix where test set only has class 2.

**v2**
Shuffled the dataset then followed same slicing method

**v3**
Added single image prediction functionality

_Check individual dataset notes_

**v4**
Applied scaling. (Jul 5 2025)



**Variables**
1. self.y_encoded -> One hot encoded values (Used on ytrain)
2. self.y_label_encoded -> Label encoded values (Used on whole Y, Must if Y contains categorical values)
3. self.labels -> If label encoder is called, it contains Classes and their assigned labels
4. self.all_prob -> Get all probabilities of a feature for being in different classes
5. self.slope -> internal variable for storing slopes


# Note
less accuracy ->
    less batch size
    less epochs
    less lr and more than enough lr
