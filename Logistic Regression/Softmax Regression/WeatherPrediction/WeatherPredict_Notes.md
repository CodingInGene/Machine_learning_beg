# Indian weather prediction


**Features** - time_epoch, time, temp_c, temp_f, is_day, wind_mph, wind_kph, wind_degree, wind_dir, pressure_mb, pressure_in, precip_mm, precip_in, humidity, cloud, feelslike_c, feelslike_f, windchill_c, windchill_f, heatindex_c, heatindex_f, dewpoint_c, dewpoint_f, will_it_rain, chance_of_rain, will_it_snow, chance_of_snow, vis_km, vis_miles, gust_mph, gust_kph, state, city

**Classes** - _Classes and their frequency_- (23 classes)
condition-
{
    'Clear': 6377, 
    'Sunny': 5895, 
    'Patchy rain possible': 3549, 
    'Light rain shower': 3670, 
    'Partly cloudy': 5126, 
    'Cloudy': 898, 
    'Moderate rain': 336, 
    'Moderate or heavy rain shower': 249, 
    'Patchy light rain': 111, 
    'Patchy light drizzle': 197, 
    'Mist': 825, 
    'Overcast': 441, 
    'Thundery outbreaks possible': 233, 
    'Moderate or heavy snow with thunder': 3, 
    'Light snow showers': 9, 
    'Patchy light snow with thunder': 3, 
    'Light drizzle': 346, 
    'Light rain': 459, 
    'Fog': 634, 
    'Moderate rain at times': 21, 
    'Heavy rain': 126, 
    'Patchy light rain with thunder': 51, 
    'Torrential rain shower': 9
}

**Dimensions** - 29568 rows x 34 columns
**Categorical columns** - "wind_dir","condition","state","city"
**Columns not used** - "wind_dir","time_epoch","time","condition","will_it_rain","chance_of_rain","will_it_snow","chance_of_snow","state","city" (_Using 24 cols_)
       ("will_it_rain","chance_of_rain","will_it_snow","chance_of_snow" these are forecast columns)

**Label encoding** - 
{'Clear': 0, 'Cloudy': 1, 'Fog': 2, 'Heavy rain': 3, 'Light drizzle': 4, 'Light rain': 5, 'Light rain shower': 6, 'Light snow showers': 7, 'Mist': 8, 'Moderate or heavy rain shower': 9, 'Moderate or heavy snow with thunder': 10, 'Moderate rain': 11, 'Moderate rain at times': 12, 'Overcast': 13, 'Partly cloudy': 14, 'Patchy light drizzle': 15, 'Patchy light rain': 16, 'Patchy light rain with thunder': 17, 'Patchy light snow with thunder': 18, 'Patchy rain possible': 19, 'Sunny': 20, 'Thundery outbreaks possible': 21, 'Torrential rain shower': 22}


**_Train set 14,784 (50%)_**

**Test1**
Models confusion matrix
       0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15    16  17  18  19
0   3008   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0    26   0   0   0
1     12   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   452   0   0   0
2      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   385   0   0   0
3      0   0   0   0   0  18   0   0   0   0   0   0   0   0   0   0    72   0   0   0
4      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   184   0   2   0
5      0   0   8   0   0  14   0   0   0   0   0   0   0   0   0   0   203   0   0   0
6     21   0  11   0   0  25   1   0   0   0   0   0   0   0   0   0  1790   0  44   0
7    174   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   243   0   4   0
8      0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   112   0   0   0
9      0   0  14   0   0  16   0   0   0   0   0   0   0   0   0   0   196   0   2   0
10     0   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0     8   0   0   0
11     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   224   0   0   0
12  1214   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  1381   0   0   0
13     7   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   108   0   9   0
14     7   0   4   0   0   9   0   0   0   0   0   0   0   0   0   0    40   0   9   0
15     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0    30   0   0   0
16    24   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  1804   0  28   0
17  2614   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0    93  54   0   0
18     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0    69   0   0   0
19     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0     3   0   0   0
Sklearns confusion matrix
       0   1    2   3   4    5    6    7   8   9   10  11    12  13  14  15    16    17  18  19
0   2267   0    0   0   0    0    0    2   0   0   0   0   168   0   0   0     0   597   0   0
1      0  46    0   0   0    0    0   84   0   0   0   0   111   0   0   0   223     0   0   0
2      0   0  197   0   0    0   59   97   0   0   0   0     0   0   0   0    32     0   0   0
3      0   2   29   0   0    7   24   18   0   0   0   2     0   0   0   0     8     0   0   0
4      0   1   72   0   0    4   46   39   0   0   0   2     0   2   0   0    20     0   0   0
5      0   2   61   0   0   31   58   43   0   0   0   7     0   0   0   0    23     0   0   0
6      0  27  107   0   0  148  998  150   3   0   0   4   139   0   0   0   312     0   4   0
7     31   5    0   0   0    0   12  189   0   0   0   0    91   0   0   0    93     0   0   0
8      0   5    0   0   0    2   74    1   0   0   0   0     8   0   0   0    24     0   0   0
9      0   4   67   0   0   22   63   46   0   0   0   5     0   0   0   0    21     0   0   0
10     0   0    0   0   0    4    3    2   0   0   0   0     1   0   0   0     2     0   0   0
11     0   5    0   0   0    0    0   53   0   0   0   8     2   0   0   0   156     0   0   0
12   121   1    0   0   0    0    0   33   0   0   0   0  2216   0   0   0    47   177   0   0
13     0   1   19   0   0    3   50   27   0   0   0   0    16   4   0   0     6     0   0   0
14     0   1   17   0   0   12   13   17   0   0   0   0     4   0   0   0     1     0   4   0
15     0   0    0   0   0    0   19    0   0   0   0   0     0   0   0   0    11     0   0   0
16     0  65   25   0   0    0  255  284   0   0   0   4    93   2   0   0  1116     1  11   0
17   631   0    0   0   0    0    0    0   0   0   0   0   140   0   0   0     0  1990   0   0
18     0   0    0   0   0    0    9    0   0   0   0   0     6   0   0   0    49     5   0   0
19     0   0    0   0   0    0    2    0   0   0   0   0     0   0   0   0     1     0   0   0
Models accuracy score 0.3301542207792208
Sklearns accuracy score 0.6129599567099567
Time taken 2.531741142272949 s
Weather Prediction of X[0] ['Patchy rain possible'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=5000,batch_size=8

**Test 2**
Models accuracy score 0.38818993506493504
Sklearns accuracy score 0.6129599567099567
Time taken 3.5447354316711426 s
Weather Prediction of X[0] ['Light rain'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=5000,batch_size=80

**Test 3**
Models accuracy score 0.5319940476190477
Sklearns accuracy score 0.6129599567099567
Time taken 18.18114185333252 s
Weather Prediction of X[0] ['Fog'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=5000,batch_size=800

**Test 4**
Models accuracy score 0.40117694805194803
Sklearns accuracy score 0.6129599567099567
Time taken 18.524901628494263 s
Weather Prediction of X[0] ['Overcast'] ,Actual ['Overcast']

With learning_rate=0.1,epochs=5000,batch_size=800

**Test 5**
Models accuracy score 0.5403814935064936
Sklearns accuracy score 0.6129599567099567
Time taken 18.405275106430054 s
Weather Prediction of X[0] ['Overcast'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=5000,batch_size=800

_Accuracy varies from 55% - 30% (randomly, on each run)_

**Test 6**
Models accuracy score 0.47307900432900435
Sklearns accuracy score 0.6129599567099567
Time taken 42.35621166229248 s
Weather Prediction of X[0] ['Fog'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=5000,batch_size=2000
_Took >42secs_

**Test 7**
Models accuracy score 0.4446022727272727
Sklearns accuracy score 0.6129599567099567
_Time taken 3.835540771484375 s_
Weather Prediction of X[0] ['Fog'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=5000,batch_size=100

_Accuracy varies from 30% - 48%, with these const params(on each run)_

**Test 8**
_Confusion matrix While 57% accuracy_
Models confusion matrix
       0   1    2   3    4   5     6   7    8   9   10  11  12    13  14  15  16   17    18   19  20
0   2008   0    0   0    0   0     0   0    1   0   0   0   0   140   0   0   0    0   881    4   0
1      0   0   39   0   73   0     0   0    9  50   0   0   0    20   0   0   0  255     3   15   0
2      0   0  339   0    2   0    42   0    0   1   0   0   0     0   0   0   0    1     0    0   0
3      0   0   55   0    2   0    25   0    0   0   0   0   1     0   0   0   0    7     0    0   0
4      0   0  118   0   11   0    45   0    0   2   0   0   0     0   0   0   0   10     0    0   0
5      0   0  127   0    9   0    75   0    0   1   0   0   0     0   0   0   0   13     0    0   0
6      0   0  150   0   37   0  1238   0   37  49   0   0   2    69   0   2   0  302     0    6   0
7      0   0    0   0    0   0     0   0    0   0   0   0   0     0   0   0   0    0     0    0   0
8     31   0  106   0   41   0     1   0  164  26   0   0   0    23   0   0   0   21     0    8   0
9      0   0    5   0    7   0    75   0    0   6   0   0   0     3   0   0   0   18     0    0   0
10     0   0  140   0    7   0    71   0    0   5   0   0   0     0   0   0   0    5     0    0   0
11     0   0    5   0    1   0     5   0    0   0   0   0   0     0   0   0   0    1     0    0   0
12     0   0   57   0   17   0     0   0    0   1   0   0   1     0   0   0   0  148     0    0   0
13   154   0    0   0   31   0     0   2   76  53   0   0   0  1598   0   0   0  197   318  166   0
14     0   0   37   0   11   0    49   0    8   7   0   0   0     4   0   2   0    4     0    4   0
15     0   0   33   0    3   0    19   0   11   1   0   0   0     2   0   0   0    0     0    0   0
16     0   0    0   0    0   0    20   0    0   0   0   0   0     0   0   0   0   10     0    0   0
17     0   0  244   0  214   0   312   0    4  95   0   0   0    24   0   0   0  917     1   45   0
18   496   0    0   0    0   0     0   0    0   0   0   0   0    98   0   0   0    1  2164    2   0
19     0   0    0   0    0   0    15   0    0   0   0   0   0     3   0   0   0   44     7    0   0
20     0   0    0   0    1   0     2   0    0   0   0   0   0     0   0   0   0    0     0    0   0

Models accuracy score 0.4876217532467532
Sklearns accuracy score 0.6129599567099567
_Time taken 4.530498743057251 s_
Weather Prediction of X[0] ['Overcast'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=8000,batch_size=100

_Accuracy varies from 41% - 57%, with these const params(on each run)_

**Test 9**
Models accuracy score 0.43702651515151514
Sklearns accuracy score 0.6129599567099567
Time taken 3.7021028995513916 s
Weather Prediction of X[0] ['Fog'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=8000,batch_size=40

**Test 10**
Models accuracy score 0.3218344155844156
Sklearns accuracy score 0.6129599567099567
Time taken 2.529369592666626 s
Weather Prediction of X[0] ['Mist'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=8000,batch_size=10

**Test 11**
Models accuracy score 0.5010822510822511
Sklearns accuracy score 0.6129599567099567
_Time taken 3.1285226345062256 s_
Weather Prediction of X[0] ['Light rain'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=10000,batch_size=10

_Accuracy varies between 41% - 50%
_Small batch size but large epochs_

**Test 12**
Models confusion matrix
       0   1   2   3    4   5     6    7   8   9   10  11    12  13  14  15    16    17  18  19
0   2231   0   0   0    0   0     0    2   0   0   0   0   357   0   0   0     0   444   0   0
1      0   0   0   0    0   0     0   28   0   0   0   0    38   0   0   0   398     0   0   0
2      0   0   0   0  164   0     0   20   0   0   0   0     0   0   0   0   201     0   0   0
3      0   0   0   0   46   0    23    0   0   0   0   0     0   0   0   0    21     0   0   0
4      0   0   0   0   69   0     0    6   0   0   0   0     0   0   0   0   111     0   0   0
5      0   0   0   0   89   0    24   12   0   0   0   0     0   0   0   0   100     0   0   0
6      0   0   0   0   28   0  1179   51   0   0   0   0    78   0   0   0   556     0   0   0
7     14   0   0   0    0   0     0  168   0   0   0   0    79   0   0   0   160     0   0   0
8      0   0   0   0    3   0    73    0   0   0   0   0     6   0   0   0    32     0   0   0
9      0   0   0   0   94   0    46   16   0   0   0   0     0   0   0   0    72     0   0   0
10     0   0   0   0    4   0     2    2   0   0   0   0     0   0   0   0     4     0   0   0
11     0   0   0   0    0   0     0   11   0   0   0   0     0   0   0   0   213     0   0   0
12    51   0   0   0    0   0     0   27   0   0   0   0  2068   0   0   0   388    61   0   0
13     0   0   0   0   37   0     3   18   0   0   0   0    10   0   0   0    58     0   0   0
14     0   0   0   0   33   0    13   16   0   0   0   0     5   0   0   0     2     0   0   0
15     0   0   0   0    0   0    16    0   0   0   0   0     0   0   0   0    14     0   0   0
16     0   0   0   0   66   0     0   85   0   0   0   0    27   0   0   0  1678     0   0   0
17   621   0   0   0    0   0     0    0   0   0   0   0   372   0   0   0     5  1763   0   0
18     0   0   0   0    0   0     0    0   0   0   0   0     7   0   0   0    61     1   0   0
19     0   0   0   0    0   0     2    0   0   0   0   0     0   0   0   0     1     0   0   0

Models accuracy score 0.5174512987012987
Sklearns accuracy score 0.6129599567099567
_Time taken 3.0947370529174805 s_

With learning_rate=0.05,epochs=12000,batch_size=10

_Accuracy varies from 27%(very low chance) - 61.93% (max accuracy given by sklearn is 61.23%)_ (Most of the time acc varies between 40% - 60%)

**Test 13**
Models accuracy score 0.5367288961038961
Sklearns accuracy score 0.6129599567099567
_Time taken 3.5199811458587646 s_
Weather Prediction of X[0] ['Overcast'] ,Actual ['Overcast']

With learning_rate=0.05,epochs=15000,batch_size=20

_Varies from 38% - 53%_

**Test 14**
Models accuracy score 0.47632575757575757
Sklearns accuracy score 0.6129599567099567
Time taken 4.496932029724121 s

With learning_rate=0.1,epochs=15000,batch_size=40

_Increased learning rate_
_Varies from 40% - 53%_


# After Scaling

**Test 15**
1. Idx = 0,
    Models accuracy score 0.8955627705627706
    Sklearns accuracy score 0.9132846320346321
    Time taken 3.8342862129211426 s
    Weather Prediction of X[0] ['Cloudy'] ,Actual ['Overcast']

    With learning_rate=0.1,epochs=12000,batch_size=40

2. Idx = 1,
    Models accuracy score 0.8580221861471862
    Sklearns accuracy score 0.9132846320346321
    Time taken 3.8253629207611084 s
    Weather Prediction of X[1] ['Overcast'] ,Actual ['Overcast']

Models confusion matrix
       0    1    2   3    4    5     6    7   8   9   10  11   12    13  14  15  16    17    18  19  20
0   2880    0    0   0    0    0     0    0   0   0   0   0    0   154   0   0   0     0     0   0   0
1      0  288    0   0    0    0    15    0   0   0   0   0   40   118   0   0   0     3     0   0   0
2      0    0  372   0   11    0     0    2   0   0   0   0    0     0   0   0   0     0     0   0   0
3      0    0    0  90    0    0     0    0   0   0   0   0    0     0   0   0   0     0     0   0   0
4      0    0    9   0  163    0     0    9   0   0   0   0    0     0   1   0   0     0     0   4   0
5      0    0    0   0    1  206     1    0   0   0   3   0    0     0   0   8   0     6     0   0   0
6      6    1    0   0    0    8  1798    0   0   0   0   0   23     1   0   4   0    49     2   0   0
7      1    0   21   0    0    0     0  389   0   2   0   0    0     0   0   0   0     0     1   7   0
8      0    0    0   1    0    1    17    0  95   0   0   0    0     0   0   0   0     0     0   0   0
9      0    0    0   0    0    0     0    0   0   0   0   0    0     0   0   0   0     0     0   0   0
10     0    0    0  66    0   64     0    0  15   0  83   0    0     0   0   0   0     0     0   0   0
11     0    0    0   5    0    0     1    0   0   0   2   0    0     0   0   2   0     2     0   0   0
12     0   47    0   0    0    0     1    0   0   0   0   0  168     6   0   0   0     2     0   0   0
13   298  168    0   0    0    0     1    0   0   0   0   0    1  2058   0   0   0     1    63   5   0
14     0    0    0   0   13    2     0    2   0   5   0   0    0     0  91   1   0     0     0  11   1
15     0    0    0   0    0   23     1    0   0   0   1   1    0     0   1  32   0    10     0   0   0
16     0    0    0   0    0    0    26    0   0   0   0   0    0     0   0   0   0     4     0   0   0
17     1   78    0   0    9   10    28    0   5   0   0   5   83    47  24   0   0  1549     2  15   0
18     0    0    0   0    0    0     0    0   0   0   0   0    0   372   0   0   0     0  2389   0   0
19     0    0    0   0    0    0     0    0   0   0   0   0    3    11   1   0   0    22     0  32   0
20     0    0    0   1    0    0     0    0   0   0   0   0    0     0   0   0   0     0     0   0   2

**Jul 6, Removed "will_it_rain","chance_of_rain","will_it_snow","chance_of_snow" columns from train set**
Current sklearn accuracy - 0.85

**Test 16**
_After removing 4 columns_

Models accuracy score 0.8361742424242424
Sklearns accuracy score 0.8523403679653679
Time taken 2.9197826385498047 s
Weather Prediction from test set X[2] ['Patchy rain possible'] ,Actual ['Cloudy']

With learning_rate=0.05,epochs=8000,batch_size=30
_Accuracy varies between 82-83%_


# Current best parameters
learning_rate=0.05,epochs=8000,batch_size=30


**_Train set 20,698 (70%), Test set 8870_**
**Test 17**
Models accuracy score 0.8453213077790305
Sklearns accuracy score 0.874859075535513
Time taken 3.0303382873535156 s

_Accuracy increased_


**Notes**
_Params -_ learning_rate=0.05,epochs=5000,batch_size=800,
_With these const params Accuracy varies from 55% - 30% (randomly, on each run)_
Also, taking above 18 secs.

1. After scaling, accuracy jumped to 80-90%. Without scaling it was 30-56%(average).

2. Batch size <80 makes alorithm fast. Bath size <20 is near stochastic, Accuracy varies too much.

3. (After scaling) learning_rate=0.05,epochs=5000,batch_size=20. _This param is good_. Execution time apprx 2.5s. Accuracy varies from 85-88%.

4. Changing **batch size to 30** (While learning_rate=0.05,epochs=5000) reduces variance in accuracy (Accuracy varies from 85-88%).

5. For real time weather prediction, algorithm was having problem to predict sunny, clear or cloudy weathers, while train set size was 14,784. 80% train set size solved this.

6. *Removing wind_degree* makes system slightly better. With wind_degree sklearn accuracy is 86.98%, without 87.01%.
7. *Removing is_day* reduces accuracy from 86.9% to 82.02%.
8. *temp_c, temp_f, pressure_mb, pressure_in, feelslike_c, feelslike_f* don't have any significant difference on accuracy individually (Slightly reduces, 86.98% to 86.94%). Means these features don't contribute that much. But together changes accuracy from 86.90% to 86.84%.
