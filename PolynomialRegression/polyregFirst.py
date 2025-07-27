import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

Y=[]    #data
X=[]

#Whole data
X=6*np.random.rand(1800,1)-3
#y=(0.8*(x**2))+(0.9*x)+2+np.random.randn(1800,1)
Y=(0.75*(X**2))+(0.75*X)-0.95+np.random.randn(1800,1)

#Training set
x=X[:800]
y=Y[:800]

#Test set
xtest=X[801:1800]
ytest=Y[801:1800]

#Transform from linear to polynomial
poly=PolynomialFeatures(degree=2)
polyXtrain=poly.fit_transform(x)
polyXtest=poly.fit(xtest)

#Training
lr=LinearRegression()
lr.fit(polyXtrain,y)

#Transform x test
x_new=np.linspace(-3,3,800).reshape(800,1)
x_new_trans=poly.transform(x_new)
y_pred=lr.predict(x_new_trans)

print(y_pred)

r2=r2_score(y,y_pred)
print("Efficiency",round(r2*100),"%")

#Plot
plt.plot(x,y,'g.',label="training set")
plt.plot(xtest,ytest,'.',label="Test set")
plt.plot(x_new,y_pred,'r',label="Prediction")
plt.legend()
plt.show()