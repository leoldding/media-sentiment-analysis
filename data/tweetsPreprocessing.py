import pandas as pd

data = pd.read_csv("tweetsOriginal.csv", delimiter=',', encoding="ISO-8859-1", header=None)

data = data.drop(columns = [i for i in range(1, 5)])

sentiment = list(data[0])

for i in range(len(sentiment)):
    sentiment[i] = 'negative' if sentiment[i] == 0 else 'neutral' if sentiment[i] == 2 else 'positive'

data[0] = sentiment

data.to_csv('tweets.csv', index = False)

print("Finished Preprocessing")