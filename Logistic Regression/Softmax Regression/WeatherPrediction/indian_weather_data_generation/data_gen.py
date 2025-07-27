#Fetch data from API and append to csv
import csv
import requests
from collections import defaultdict
import numpy as np
import pandas as pd
import threading

#Function block
def data_gen(city,order_features,order_target):
    api_key = "c1926255a75c40e799564438250507"
    api_url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={city}&days=1&aqi=no&alerts=no"  #api url for forecast

    response = requests.get(api_url)
    data = response.json()
    
    loc = data["location"]  #Contains regional info
    forecast_all = data["forecast"]
    forecast_day = forecast_all["forecastday"][0]
    hours = forecast_day["hour"]    #Gives array of dicts - [ {1st hr forecast}, {2nd hr forecast} ...]

    #Data of all features
    feature_data=[]    #Contains lists of every hour data in specific format

    for eachhour in hours:  #Travese hours data
        temp=[]
        for ele in order_features:   #Use predefined format
            temp.append(eachhour[ele])
        feature_data.append(temp)

    #Data of all targets
    target_data=[]

    for eachhour in hours:  #Travese hours data
        temp=[]
        for ele in order_target:   #Use predefined format
            temp.append(eachhour[ele])
        target_data.append(temp)
    #Extracting condition value ({'text': 'Light rain shower', 'icon': '//cdn.weatherapi.com/weather/64x64/night/353.png', 'code': 1240})
    for ele in target_data:
        cond = ele[0]
        ele[0] = cond["text"]

    return feature_data, target_data, loc #Return features, targets, location

def doExists(data): #Checks if same data exists already
    weather = pd.read_csv("indian_weather_2025.csv").to_numpy()
    
    #Converting to list as 'i in a' can't be done in np array
    a=weather.tolist()
    b=data.tolist()
    
    for i in b:
        if i in a:
            return True
    return False
    

def addToCsv(feature_data, target_data, location):
    city = location["name"]
    state = location["region"]
    time_epoch = location["localtime_epoch"]
    localtime = location["localtime"]

    #Data formatting to existing header format
    all_data = np.zeros(len(feature_data), dtype="object")

    for i in range(0,len(feature_data)):
        all_data[i] =  feature_data[i] + target_data[i] + [location["region"],location["name"]]
    all_data = np.array(all_data)

    #Check if same data exists
    if not doExists(all_data):
        file = open("indian_weather_2025.csv","a",newline='')
        writer = csv.writer(file)
        writer.writerows(all_data)

        #print(all_data)
        print("Data added to CSV",city)
    else:
        print("Duplicate data present for",city)

def processExecutor(cities,order_features,order_target):
    for i in cities:
        #Generate data
        feature_data, target_data, location = data_gen(i,order_features,order_target)

        #Add to CSV file
        addToCsv(feature_data, target_data, location)


#Main block
if __name__=="__main__":
    #Total 29 cities
    cities = ["Kolkata","Durgapur","Bardhaman","Chinsurah","Bagdogra","Dhulian","Howrah","Bangalore","New Delhi","Junagadh","Jamnagar","Agra","Shillong","Mawsynram","Cherrapunji","Mahabaleshwar","Pasighat","Sitarganj","Gangtok","Puri","Digha","Surat","Pondicherry","Udupi","Chennai","Jaipur","Udaipur","Ajmer","Darjeeling"]

    order_features = ['time_epoch', 'time', 'temp_c', 'temp_f', 'is_day', 'wind_mph', 'wind_kph', 'wind_degree', 'wind_dir', 'pressure_mb', 'pressure_in', 'precip_mm', 'precip_in', 'humidity', 'cloud', 'feelslike_c', 'feelslike_f', 'windchill_c', 'windchill_f', 'heatindex_c', 'heatindex_f', 'dewpoint_c', 'dewpoint_f', 'vis_km', 'vis_miles', 'gust_mph', 'gust_kph']
    order_target = ['condition', 'will_it_rain', 'chance_of_rain', 'will_it_snow', 'chance_of_snow']

    # for i in cities:
    #     #Generate data
    #     feature_data, target_data, location = data_gen(i,order_features,order_target)

    #     #Add to CSV file
    #     addToCsv(feature_data, target_data, location)
    t1=threading.Thread(target=processExecutor, args=(cities[:7],order_features,order_target))
    t2=threading.Thread(target=processExecutor, args=(cities[7:14],order_features,order_target))
    t3=threading.Thread(target=processExecutor, args=(cities[14:21],order_features,order_target))
    t4=threading.Thread(target=processExecutor, args=(cities[21:29],order_features,order_target))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()