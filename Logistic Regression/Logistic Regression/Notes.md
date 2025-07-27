# Iris dataset

**feture_names** - 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)' *( x1,x2,x3,x4 )*
**target_names** - 'setosa' 'versicolor' 'virginica' *( 0,1,2 )*
**target** - y values (0-2)
**data** - x values 4 features.

*So,*
    If classifying for setosa then if 0 then 1 means setosa otherwise 0 not setosa. *(setosa is 0 so if y=0 then setosa, 1 versicolor, 2 virginica)*

# Notes
Data points 100-
    Best at lr=0.05, epochs=8000

Data points 200
    Has outlier, which remains unclassified