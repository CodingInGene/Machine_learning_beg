import pandas as pd
import numpy as np

#Function block
def loadData():
    #Play tennis dataset (Depending on the weather play tennis or not)
    tennis = pd.read_csv("play_tennis.csv")
    x=np.transpose(np.array([tennis["outlook"],tennis["temp"],tennis["humidity"],tennis["wind"]]))
    y=np.array(tennis["play"])
    return x,y


#Main block
if __name__=="__main__":
    X,Y = loadData()
    print(X,Y)