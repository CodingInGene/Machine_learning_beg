#Create SVM classifier for linear dataset
from sklearn.svm import SVC
from sklearn.datasets import make_classification,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd
import numpy as np

#Function block
def loadData(dataset):
    if dataset == "persona":
        persona = pd.read_csv("personality_datasert.csv")
        x=np.transpose( [ persona["Time_spent_Alone"].to_numpy(), persona["Friends_circle_size"].to_numpy() ] )
        
        label = LabelEncoder()
        y = label.fit_transform(persona["Personality"])
        return x,y
    if dataset == "random":
        x,y=make_classification(n_samples=500,n_features=2,n_informative=1,n_redundant=0,n_classes=2,n_clusters_per_class=1,class_sep=15,hypercube=False,random_state=41)
        return x,y


#Main block
if __name__=="__main__":
    X,Y = loadData(dataset="random")

    xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.5)
    print(xtrain.shape,ytrain.shape)

    #Training
    c=100
    svc=SVC(C=c,kernel="linear")

    #Prediction
    svc.fit(xtrain,ytrain)
    print("Weights",svc.intercept_,svc.coef_)

    y_pred = svc.predict(xtest)

    #Decision boundary
    support_vectors_indices = svc.support_
    print("Support vectors",support_vectors_indices)

    support_vectors = svc.support_vectors_
    
    #Plot
    fig,ax = plt.subplots(nrows=2)
    ax[0].scatter(xtrain[:,0],xtrain[:,1],c=ytrain,cmap="winter")

    ax[1].scatter(xtrain[:,0],xtrain[:,1],c=ytrain,cmap="winter")
    ax[1].scatter(support_vectors[:,0],support_vectors[:,1],c="r")

    common_params = {"estimator":svc,"X":xtest,"ax":ax[1]}
    DecisionBoundaryDisplay.from_estimator(**common_params,response_method="predict",plot_method="pcolormesh",alpha=0.3,grid_resolution=200)
    DecisionBoundaryDisplay.from_estimator(**common_params,response_method="decision_function",plot_method="contour",levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    ax[0].set_title("Random dataset")
    ax[1].set_title(f"SVM classification, C = {c}")
    plt.show()