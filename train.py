import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import functions
import sys
import time
# read in data to dataframe
data = pd.read_csv("data/" + sys.argv[1] + ".csv", delimiter=',', encoding="ISO-8859-1", header=None, names=["sentiment", "headlines"])

# create train and test data splits
train, test = train_test_split(data, test_size=0.10, random_state=42)

# create the csv for testing
test.to_csv(sys.argv[1] + 'test.csv', index=False)

# number of splits in k-fold cross validation
splits = 10

# variables used to keep track of optimal parameters
minRMSE = float('inf')
bestFlip = 0
bestDiminish = 0

# parameters to test
flipParams = [1, 2, 3]
diminishParams = [i for i in np.arange(0.05, 1, 0.05)]

original_time = time.time()

# grid search cv
for flip in flipParams:
    for diminish in diminishParams:
        start_time = time.time()
        print("Parameters: {flip = " + str(flip) + ", diminish = " + str(diminish) + "}")
        avgRMSE = 0
        kfold = KFold(n_splits=splits, shuffle=True, random_state=42).split(train)  # create k fold splits
        for train_index, validation_index in kfold:
            word_sentiments = functions.assignSentiment(flip, diminish, train.iloc[train_index])  # train system
            avgRMSE += functions.calculateRMSE(word_sentiments, train.iloc[validation_index])  # validate system
        avgRMSE /= splits
        # check if parameters performed the best thus far
        if avgRMSE < minRMSE:
            # update optimal parameters
            minRMSE = avgRMSE
            bestFlip = flip
            bestDiminish = diminish
        print(time.time() - start_time)
        print("Average RMSE: " + str(avgRMSE) + '\n')

print("Total Time: " + str(time.time() - original_time) + " seconds")
print("Best Flip Value: " + str(bestFlip))
print("Best Diminish Value: " + str(bestDiminish))
# train final system based on optimal parameters
word_sentiments = functions.assignSentiment(bestFlip, bestDiminish, train)

# output word sentiment values to text file
with open(sys.argv[1] + 'sentiments.txt', 'w') as f:
    for word in word_sentiments.keys():
        if word_sentiments[word] != 0:
            f.write(word + ' ' + str(word_sentiments[word]) + '\n')
