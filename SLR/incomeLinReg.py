import matplotlib.pyplot as plt
import csv
from lin_regression import getData

x=[]
y=[]
#File
file=[]
f=open("income.data.csv",'r')
csv=csv.reader(f)    #csv.reader returns every row as list
for row in csv:
    file.append(row)
#Training data
for data in range(1,80):
    x.append(round(float(file[data][1]),3))    #String to float
    y.append(round(float(file[data][2]),3))      #Setting x & y training data

#Linear properties find
m,c=getData(x,y)
print("M: ",m,"C: ",c)

#Prediction
inpX=[]
actual=[]
count=0
for i in range(80,len(file)-1):
    ind=round(float(file[i][1]),3)
    inpX.append(ind)
    x.append(ind)
    res=(m*ind)+c   #Y prediction
    y.append(round(res,3))
    actual.append(round(float(file[i][2]),3))   #Actual Y value

    if res!=round(float(file[i][2]),3):         #Counting outliers
        count+=1

#print("Total outliers ",count)	#Not outliers
#Sorting x & y
#x.sort()
#y.sort()   #Not doing sort is obvious
#Graph plotting
plt.scatter(x,y,facecolor="orange",marker=".")
plt.scatter(inpX,actual,marker=".")
plt.xlabel("Income(x$10,000)")
plt.ylabel("Happieness(0-10)")
plt.title("Income vs Happieness prediction through Linear regression")
plt.show()
