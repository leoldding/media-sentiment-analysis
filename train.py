import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import functions

# read in data to dataframe
data = pd.read_csv("all-data.csv", delimiter=',', encoding="ISO-8859-1", header=None, names=["sentiment", "headlines"])

# create train and test data splits
train, test = train_test_split(data, test_size=0.10, random_state=42)

test.to_csv('test.csv', index=False)

splits = 10

minRMSE = float('inf')
bestFlip = 0
bestDiminish = 0
flipParams = [1, 2, 3]
diminishParams = [i for i in np.arange(0.05, 1, 0.05)]

for flip in flipParams:
    for diminish in diminishParams:
        print("Parameters: {flip = " + str(flip) + ", diminish = " + str(diminish) + "}")
        avgRMSE = 0
        kfold = KFold(n_splits=splits, shuffle=True).split(train)
        for train_index, validation_index in kfold:
            word_sentiments = functions.assignSentiment(flip, diminish, train.iloc[train_index])
            avgRMSE += functions.calculateRMSE(word_sentiments, train.iloc[validation_index])
        avgRMSE /= splits
        if avgRMSE < minRMSE:
            minRMSE = avgRMSE
            bestFlip = flip
            bestDiminish = diminish
        print("Average RMSE: " + str(avgRMSE) + '\n')

print("Best Flip Value: " + str(bestFlip))
print("Best Diminish Value: " + str(bestDiminish))
word_sentiments = functions.assignSentiment(bestFlip, bestDiminish, train)

# normalize and output word sentiment values
with open('sentiments.txt', 'w') as f:
    for word in word_sentiments.keys():
        f.write(word + ' ' + str(word_sentiments[word]) + '\n')