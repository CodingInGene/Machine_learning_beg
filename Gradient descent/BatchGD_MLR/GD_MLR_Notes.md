intercept = 0, all coeffs = 1

**With sklearn's inbuit diabetes dataset**
total rows 442, 10 cols
xtrain = ytrain = 100

1. at learning rate = 0.5, epochs = 800 - r2 score 0.477
2. at learning rate = 0.5, epochs = 400 - r2 score 0.440
3. at learning rate = 0.5, epochs = 1500 - r2 score 0.471

xtrain = ytrain = 200
4. at learning rate = 0.5, epochs = 800 - r2 score 0.506

**Best possible settings is lr 0.5, epochs 800**


**With housing dataset**
features - housing_median_age, population

1. at learning rate = 1*10^6, epochs = 2000 - r2 score -0.56
2. at learning rate = 1*10^5, epochs = 2000, intercept and coeffs are having nan values. Same with 2*10^6 and 5*10^6
3. at learning rate = 1*10^6, epochs = 10,000 - r2 score -0.558

    Possible cause, dataset is not scaled

features - housing_median_age, total_rooms

1. at learning rate = 1*10^6, epochs = 10,000, having nan
2. at learning rate = 1*10^7, epochs = 10k - r2 score -0.431
3. at learning rate = 1*10^7, epochs = 12k - r2 score -0.413
