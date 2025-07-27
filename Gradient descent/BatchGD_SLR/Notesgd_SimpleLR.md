# Outcomes

**from incomeData_SLR.csv**

training set - 80 (x,y each)

Solution b=0.1497 m=0.7309
1. Set 1
    m=120
    b=-120

    Possible solution b=0.14975 m=0.7309 (_Got from sklearn_)

    at epochs=6030 converging to solution(closest). 6500 epochs adds error to b +0.001 (b=0.149) but m is same(+0.0000001 error)

2. Set 2
    m=0
    b=0
    lr=0.0001   (on 0.001 it jumps too arbitarily, and never converges to solution, at around 20 epochs it reaches >10^6 on b)
    
    at epochs 180 m=0.7320, b=0.1443
    at epochs 400 m=0.7316, b=0.1462
    at epochs 800 m=0.7312, b=0.1481
    at epochs 2500 m=0.73099, b=0.14969     **Closest**

    After 500 plot is flat

3. Set 3
    m=0
    b=0
    lr=0.0005

    at epochs 1000 m=0.73098, b=0.14975     **Closest to the optimal solution**

**Notes**
The equation for slope = y[i] - m*x[i] - b ,is true but in python it must be written - y[i] - (m*x[i] + b) because of precendence, to remove any ambiguity.

**Plot animation of prediction line at different m, b values**
1. make a 2d array(y_pred_lines) and on each row store the line obtained from prediction function at each b,m until it converges to the solution
2. Then use xtest(test x data) and y_pred_lines to animate
3. Slow down the animation. (make interval bigger, e.g-100,150)
4. For saving use ffmpeg writer. If not installed - **pip install ffmpeg.**

def animator(frame):    **Animate the plot of xtrain and y_pred lines**
    animated_plot.set_data(xtest,y_pred_lines[frame])
    return animated_plot,

y_pred_lines=[]
for i in range(0,len(change_in_b)):
    y_pred_lines.append(gdPredict(xtest,change_in_b[i],change_in_m[i]))

fig,axis=plt.subplots()
axis.plot(xtest,ytest,'.')
axis.set_xlim([1,8])
axis.set_ylim([0,7])
animated_plot,=axis.plot([],[])

animation=FuncAnimation(
    fig=fig,
    func=animator,
    frames=len(y_pred_lines),
    interval=150,    #millisec
    repeat=True,
)
animation.save("convergence.mp4",writer="ffmpeg",fps=5)
print("Animation saved")
plt.show()