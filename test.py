import pandas as pd
import math
import random

# read in test dataset
test = pd.read_csv('test.csv')

# create dictionary for word-sentiment pairs
word_sentiments = {}

# read in word-sentiment pairs from train.py output
with open('sentiments.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        split = line.split(' ')
        word_sentiments[split[0]] = float(split[1])

# initialize rmse variables
system_rmse = 0
baseline_rmse = 0

# set random seed for baseline
random.seed(42)

# iterate through test set
for i in range(len(test)):
    true_sentiment = 1 if test.sentiment.iloc[i] == 'positive' else -1 if test.sentiment.iloc[i] == 'negative' else 0
    headline = test.headlines.iloc[i].lower().split(' ')

    sentiment_score = 0
    sentiment_abs = 0  # used to standardize score between -1 and 1

    # iterate through words in headline incrementing score based on training dictionary
    for word in headline:
        if word in word_sentiments:
            sentiment_score += word_sentiments[word]
            sentiment_abs += abs(word_sentiments[word])

    # standardize score
    if sentiment_abs != 0:
        sentiment_score /= sentiment_abs

    # increment system rmse
    system_rmse += abs(true_sentiment - sentiment_score) ** 2

    # choose random sentiment and incresase baseline rmse
    random_sentiment = random.randint(-1, 1)
    baseline_rmse += abs(true_sentiment - random_sentiment) ** 2

# calculate normalized rmse
normalized_system_rmse = math.sqrt(system_rmse / len(test)) / 2
normalized_baseline_rmse = math.sqrt(baseline_rmse / len(test)) / 2

print("Normalized System RMSE: " + str(normalized_system_rmse))
print("Normalized Baseline RMSE: " + str(normalized_baseline_rmse))
