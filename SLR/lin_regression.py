#Linear regression
import matplotlib.pyplot as plt

def mean(var):
    s=0
    for i in var:
        s+=i
    mean=s/len(var)
    return mean

def getData(x,y):
    #Mean
    mx=mean(x)
    my=mean(y)
    #Deviation
    dvx=[]          #Deviation of x
    for i in x:
        dvx.append(i-mx)
    dvy=[]          #Deviation of y
    for i in y:
        dvy.append(i-my)
    #Product of x,y deviations
    pd=[]
    for i in range(len(dvx)):
        pd.append(dvx[i]*dvy[i])
    #SOP of x,y deviations
    sopd=0
    for i in pd:
        sopd+=i
    #Square deviation of x
    sqdx=[]
    sumdx=0 #Sum of sqdx
    for i in dvx:
        sqdx.append(i**2)
        sumdx+=i**2
    #Slope
    m=sopd/sumdx
    #Intercept
    c=my-(m*mx)
    return m,c

def lin(x,m,c):
    y=(m*x)+c
    return y

if __name__=='__main__':
    x=[]
    y=[]
    #Data Input phase
    n=int(input("Enter range of inputs to feed: "))
    for i in range(n):
        x.append(int(input("Enter x: ")))
        y.append(int(input("Enter y: ")))
        print()

	#Numericals
    m,c=getData(x,y)
    print("M ",m,"C ",c)

	#Now Prediction
    inpX=[]
    actY=[] #Actual y value
    n=int(input("Range of predictions: "))
    for i in range(n):
        inp=int(input("Enter x: "))
        inpX.append(inp)    #Input x
        x.append(inp)       #Extending x
        res=lin(inp,m,c)
        y.append(res)       #Extending y
        print("Y should be: ",res)
	    #Actual value input
        ay=float(input("Actual y value(float):"))
        actY.append(ay)

	#Graph plotting
    print(inpX)
    print(actY)
    plt.plot(x,y)
    plt.plot(inpX,actY,'.')
    plt.xlabel("X-->")
    plt.ylabel("Y-->")
    plt.show()
