#Softmax regression
import numpy as np
from sklearn.datasets import load_iris,fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as ttsp
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib import pyplot as plt
import time
from collections import defaultdict
import requests

#Function block
def loadData(dataset):
    if dataset=="weather":
        weather=pd.read_csv("indian_weather_forecast_data.csv")
        print(weather)
        x = weather.drop(columns=["wind_dir","time_epoch","time","condition","will_it_rain","chance_of_rain","will_it_snow","chance_of_snow","state","city"]).to_numpy()
        y = weather["condition"].to_numpy()

        # '''Adding wind_dir to features'''
        # '''Adding it, doesn't give any benifit'''
        # wind_dir = weather["wind_dir"]
        # #Label encoding
        # softmax = Softmax()
        # softmax.labelencoder(np.transpose([wind_dir.to_numpy()]))
        # y_wind = softmax.y_label_encoded
        # # print(softmax.y_label_encoded,softmax.labels)
        # #Adding it to features
        # x = np.insert(x,obj=6,values=np.transpose(y_wind)[0],axis=1)    #we cannot add y_wind in Nx1 matr in axis=1 properly. It has to be scalar
        
        print(x[0],y)

    return x,np.transpose([y])

def traintestsplit(x,y,trainrows):
    #return [x[np.random.randint(0,x.shape[0])] for i in range(0,trainrows)],y[:trainrows],x[trainrows:],y[trainrows:]
    # np.random.shuffle(x)  #Shuffling will worsen this model
    # np.random.shuffle(y)
    return x[:round(x.shape[0]*trainrows)],y[:round(x.shape[0]*trainrows)],x[round(x.shape[0]*trainrows):],y[round(x.shape[0]*trainrows):]

def currentweatherdetails(city):    #Returns list of weather data and dict of location and time details
    '''Takes real time data from Weather API'''
    #Predefined key order of weather data
    weather_data = {
        "temp_c": 32,
        "temp_f": 89.6,
        "is_day": 1,
        "wind_mph": 6.83,
        "wind_kph": 11,
        "wind_degree": 213.01,
        "pressure_mb": 998,
        "pressure_in": 29.47,
        "precip_mm": 2.5,
        "precip_in": 0.098,
        "humidity": 87,
        "cloud": 100,
        "feelslike_c": 39,
        "feelslike_f": 102.2,
        "windchill_c": 28,
        "windchill_f": 82.4,
        "heatindex_c": 37,
        "heatindex_f": 98.6,
        "dewpoint_c": 27,
        "dewpoint_f": 80.6,
        "vis_km": 6,
        "vis_miles": 3.72,
        "gust_mph": 16.15,
        "gust_kph": 26
    }
    # Api - http://api.weatherapi.com/v1/current.json?key={{apikey}}&q={{city}}&aqi=no
    api_key = "xx..xxxx"
    api_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    
    response = requests.get(api_url)
    data = response.json()

    current = data["current"]
    location = data["location"]
    
    #Convert json to list
    inp = []
    for i in weather_data.keys():
        inp.append(current[i])
        
    return np.insert( np.array(inp), obj=0,values=1 ) , location    #Added bias term

def softmaxfunc(z):
    if len(z.shape) > 1:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    else:   #For single image
        exp_z = np.exp(z - z.max())
        return exp_z / np.sum(exp_z)

class Softmax:
    def __init__(self):
        self.y_encoded=None
        self.y_label_encoded=None
        self.labels=None
        self.epochs=None
        self.lr=None
        self.weights=None
        self.kroneckerDelta=None   #Not used
        self.slope=None
        self.y_pred=None
        self.all_prob=None
    def hotencoder(self,y):    #One hot encoding, takes Nx1 matrix
        #Finding unique classes
        no_of_classes = len(np.unique(y))
        self.y_encoded=np.zeros([y.shape[0],no_of_classes])

        #Assign labels to each class
        n=np.unique(y)
        d=defaultdict(int)
        count=0
        for i in np.unique(y):
            d[i] = count
            count+=1

        #Encode
        for i in range(0,self.y_encoded.shape[0]):
            label = d[ y[i][0] ]
            self.y_encoded[i][label] = 1
    def labelencoder(self,y):
        self.y_label_encoded = []

        #Assign labels to each class
        n=np.unique(y)
        d=defaultdict(int)
        count=0
        for i in np.unique(y):
            d[i] = count
            count+=1

        for i in range(0,y.shape[0]):
            self.y_label_encoded.append( d[y[i][0]] )
        
        self.labels = d     #Categorical to labels

        self.y_label_encoded = np.transpose( [np.array(self.y_label_encoded)] )
    def labeldecoder(self,y):
        decoded = []
        for i in y:
            for key,val in self.labels.items():
                if val == i:
                    decoded.append(key)
        return decoded
    def fit(self,x,learning_rate,epochs,batch_size):
        self.epochs=epochs
        self.lr=learning_rate
        self.weights=np.zeros([y_encoded.shape[1],x.shape[1]])  # no of classes x no of features+1

        #Kronecker delta function
        '''
        Making Kronecker delta matrix by shape of X dot Weights -> Nx(m+1) (where m = no of classes), as it's dim are same as y_hat
        Filling diagonals with 1, non diagonals with 0
        '''
        # self.kroneckerDelta = np.zeros(y_encoded.shape)
        # for i in range(0,x.shape[0]):
        #     for j in range(0,x.shape[1]):
        #         if i==j:
        #             self.kroneckerDelta[i][j]=1
        
        for i in range(0,self.epochs):
            batch=np.array( [np.random.randint(0,x.shape[0]) for i in range(batch_size)] )
            #print(x[batch])

            y_hat = softmaxfunc(np.dot(x[batch],np.transpose(self.weights)))
            '''
            self.slope = np.dot(np.dot( np.transpose(self.y_encoded), np.transpose(np.transpose(self.kroneckerDelta) - y_hat) ), x)
            Formula -> slope=Y.[Kronecker - Y_hat].X
            Not using kronecker because we are using vector form, kronecker is used in derivation of loss func.
            Adding kronecker here causes output to Nx(m+1)
            Now formula is (Y-y_hat)*X
            '''
            self.slope = np.dot(np.transpose(y_encoded[batch] - y_hat),x[batch])
            self.weights += self.lr * self.slope

    def predict(self,xtest):    #Takes Nx(m+1)
        self.all_prob = softmaxfunc(np.dot(xtest,np.transpose(self.weights)))    #Probability of all classes
        if len(self.all_prob.shape) > 1:
            self.y_pred = np.argmax(self.all_prob,axis=1)   #Get which col(class) has max probability
        else:   #For single image
            self.y_pred = np.argmax(self.all_prob)


#Main block
if __name__=="__main__":
    start_time=time.time()

    dataset="weather"
    X,Y=loadData(dataset=dataset)

    #Scaling
    std = StandardScaler()
    X = std.fit_transform(X)

    softmax=Softmax()

    #Label encoding
    '''Label encoding Y as in weather data Condition contains categorical data. Must, if Y contains categorical values'''
    softmax.labelencoder(Y)
    Y_new = softmax.y_label_encoded

    #Train test split
    xtrain,ytrain,xtest,ytest = traintestsplit(X,Y_new,trainrows=0.8)


    #Change X for bias term
    xtrain_new = np.insert(xtrain,obj=0,values=1,axis=1)
    xtest_new = np.insert(xtest,obj=0,values=1,axis=1)
    print(xtrain_new.shape,ytest.shape)

    #One hot encoding
    softmax.hotencoder(ytrain)
    y_encoded=softmax.y_encoded
    #print(y_encoded)

    #Training
    softmax.fit(xtrain_new,learning_rate=0.05,epochs=9000,batch_size=30)
    #print("Models prediction weight\n",softmax.weights)

    #Prediction
    softmax.predict(xtest_new)
    #print("Models y predicted",softmax.y_pred)

    '''
    Sklearn
    '''
    #Sklearns logistic regression
    logit = LogisticRegression()
    logit.fit(xtrain,np.transpose(ytrain)[0])
    #print("\nSklearns weights","Intercept",logit.intercept_,"Coef\n",logit.coef_)
    sk_pred = logit.predict(xtest)
    #print("Sklearns y predicted",sk_pred)

    #Error metrics
    model_conf = confusion_matrix(ytest,softmax.y_pred)
    acc_score=accuracy_score(ytest,softmax.y_pred)
    print("Models confusion matrix\n",pd.DataFrame(model_conf))

    sk_conf = confusion_matrix(ytest,sk_pred)
    sk_acc=accuracy_score(ytest,sk_pred)
    print("Sklearns confusion matrix\n",pd.DataFrame(sk_conf))

    #print("Labels",softmax.labels)  #Print encoded labels
    
    print("Models accuracy score",acc_score)
    print("Sklearns accuracy score",sk_acc)

    end_time=time.time()
    print("Time taken",end_time-start_time,"s")


    #Weather prediction
    idx = 34
    softmax.predict(xtest_new[idx])     # [ x1,x2,...,xn ] 1xM

    #Decode label encoded y
    decoded_pred = softmax.labeldecoder( [softmax.y_pred] )
    decoded_actual = softmax.labeldecoder( [ytest[idx]] )

    print(f"Weather Prediction from test set X[{idx}] {decoded_pred} ,Actual {decoded_actual}")
    print(f"Probability of prediction {np.max(softmax.all_prob)*100}%")

    #Real time weather prediction
    switch = 1  # 1 - Use API, 0 - No real time prediction

    if switch == 1:
        #ci = input("\nEnter city or region: ")
        data, location = currentweatherdetails(city="Chinsurah")
        #data=np.array([30.1, 86.2, 1, 6.7, 10.8, 9, 1011.0, 29.85, 0.0, 0.0, 40, 0, 27.9, 82.3, 33.3, 92.0, 31.3, 88.4, 7.8, 46.0, 10.0, 6.0, 7.7, 12.4])
        pred = logit.predict(np.delete(data,obj=0).reshape(1,-1))
        print(pred)

        softmax.predict(data)
        decoded_pred_1 = softmax.labeldecoder( [softmax.y_pred] )
        
        localtime = location["localtime"]
        lat = location["lat"]
        long = location["lon"]
        c_name = location["name"]
        region = location["region"]
        print(f"\n{localtime}, Latitude - {lat}, Longitude - {long}, Region - {c_name}, {region} \nReal time Weather Prediction result: {decoded_pred_1}")
        print(f"Probability of prediction {np.max(softmax.all_prob)*100}%")


'''
Note -
Use of important variables are mentioned in Softmax_notes (Main notes)
'''
