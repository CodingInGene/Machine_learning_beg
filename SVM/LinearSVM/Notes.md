# Dataset - personality_dataset, Kaggle

Rows 2900
Features - 7    (X=7, Y=1)
Classes - 2 **Extrovert(0) or Introvert(1)**

**On def loadData->**
x=np.transpose( [ persona["Time_spent_Alone"].to_numpy(), persona["Friends_circle_size"].to_numpy() ] )

*to_numpy* - converts pandas dataframe to numpy array
1. [ x1, x2]
2. then transpose


# Notes
With higher C value margin width is increasing