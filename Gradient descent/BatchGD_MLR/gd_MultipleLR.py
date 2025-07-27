#Perform MLR on housing dataset with Gradient Descent
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import math
from scipy.interpolate import interp1d

#Function block
def getData():
    X,Y = load_diabetes(return_X_y=True)    #Sklearn's inbuit dataset
    return X,np.transpose([Y])

    # housing=pd.read_csv("housing.csv")      #Read data
    # housing.dropna(how="any",inplace=True)  #Remove null rows

    # #Make X
    # X1=[]
    # for i in housing["housing_median_age"]:
    #     X1.append(i)
    # X2=[]
    # for i in housing["total_rooms"]:
    #     X2.append(i)
    # X3=[]
    # for i in housing["median_income"]:
    #     X3.append(i)
    
    # X=np.transpose([X1,X2,X3])         # n x 1 matrix

    # #Make Y
    # Y_0=[]
    # for i in housing["median_house_value"]:
    #     Y_0.append(i)
    # Y=np.transpose([Y_0])           # n x 1 matrix

    return X,Y

def gdRegressor(x,y,learning_rate=0.01,epochs=100):
    lr=learning_rate
    intercept=0
    coeffs=np.ones((x.shape[1],1))

    for i in range(0,epochs):
        y_hat = np.dot(x,coeffs) + intercept    # y hat Nx1 , x dot coeffs -> Nx1
        slope_intercept = np.mean(y-y_hat) * (-2)   #slope of b0 = -2 * mean(y[i] - y_hat[i])
        slope_coeffs = np.dot( np.transpose(x), (y-y_hat) ) * (-2/x.shape[0])  # y-y_hat is [n,1], x is [n,2], so transpose(x) -> [2,n]. so x*(y-y_hat) -> [2,n] dot [n,1] = [2,1]

        intercept = intercept - (lr*slope_intercept)
        coeffs = coeffs - (lr*slope_coeffs)      # subtracting coeffs 1xN matrix with scalar slope -> matrix 1xN

        if math.isnan(intercept):
            break

    return intercept,np.transpose(coeffs)    #Coeff is 2x1 matrix so transpose is 1x1

def gdPredict(x,intercept,coeffs):
    y=np.dot(x,np.transpose(coeffs)) + intercept
    return y


#Main block
if __name__=="__main__":
    X,Y = getData()
    # print("X",X,"Y",Y)
    print("Shape of X",X.shape,"Shape of Y",Y.shape)

    #Split training and test data
    xtrain=X[:300]
    ytrain=Y[:300]

    xtest=X[301:]
    ytest=Y[301:]

    #Calculating feature weights from sklearn
    lreg=LinearRegression()
    lreg.fit(xtrain,ytrain)
    print("Sklearn intercept",lreg.intercept_,"Coeffs",lreg.coef_)

    #Getting feature weights
    learning_rate=0.5
    epochs=800
    intercept,coeffs=gdRegressor(xtrain,ytrain,learning_rate=learning_rate,epochs=epochs)    #Diabetes data
    #intercept,coeffs=gdRegressor(xtrain,ytrain,learning_rate=0.0000001,epochs=12000)

    print("\nModel intercept",intercept,"Coeffs",coeffs)

    #Predictions
    y_pred=gdPredict(xtest,intercept,coeffs)    #Pass the coeff as 1x1 - [[b1,b2,...,bm]]

    r2=r2_score(ytest,y_pred)
    print("\nModels r2 score",r2)

    #Plot
    x_new=np.transpose(xtest)[0]
    # plt.scatter(x_new,ytest)
    # plt.scatter(x_new,y_pred)
    # plt.show()
    sort_idx = np.argsort(x_new)
    x_sorted = x_new[sort_idx]
    y_true_sorted = ytest[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # Plot actual vs predicted
    plt.scatter(x_sorted, y_true_sorted, color='blue', label='Actual')
    plt.scatter(x_sorted, y_pred_sorted, color='red', label='Predicted', alpha=0.5)
    
    # Create regression line using interpolation
    reg_line = interp1d(x_sorted.flatten(), y_pred_sorted.flatten(), kind='linear')
    x_line = np.linspace(x_sorted.min(), x_sorted.max(), xtest.shape[0])
    plt.plot(x_line, reg_line(x_line), color='green', linewidth=2, label='Regression Line')
    
    # Add sklearn's regression line for comparison
    sklearn_pred = lreg.predict(xtest)[sort_idx]
    plt.plot(x_sorted, sklearn_pred, color='orange', linestyle='--', label='Sklearn Regression')

    # c=interp1d(np.transpose(xtest)[0],np.transpose(y_pred)[0],kind="linear")
    # x_smooth=np.linspace(np.transpose(xtest)[0].min(),np.transpose(xtest)[0].max(),xtest.shape[0])
    # y_smooth=c(x_smooth)
    # plt.plot(x_smooth,y_smooth)
    
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title(f'MLR using Gradient Descent\nLearning rate {learning_rate} Epochs {epochs}, R2 score {r2}')
    plt.legend()
    plt.show()
