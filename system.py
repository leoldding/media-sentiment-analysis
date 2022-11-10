import pandas as pd
from sklearn.model_selection import train_test_split
import math
import random
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# things to avoid
punctuation_and_numbers = ["!", "@", "#", "$", "%", "'", "^", "&", "*", "(", ")", "{", "}", "[", "]", "\\", "|", "=",
                           "+", "/", "?", "-", "_", ".", "<", ">", "`", "~", ";", ":", ",",
                           '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# many more things to avoid
stop_words = ['a', 'the', 'an', 'and', 'or', 'but', 'about', 'above', 'after', 'along', 'amid', 'among',
              'as', 'at', 'by', 'for', 'from', 'in', 'into', 'like', 'near', 'of', 'off', 'on',
              'onto', 'out', 'over', 'past', 'per', 'plus', 'since', 'till', 'to', 'under', 'until', 'up',
              'via', 'vs', 'with', 'that', 'could', 'may', 'might', 'must',
              'need', 'ought', 'shall', 'should', 'will', 'would', 'have', 'had', 'has', 'having', 'be',
              'is', 'am', 'are', 'was', 'were', 'being', 'been', 'get', 'gets', 'got', 'gotten',
              'getting', 'seem', 'seeming', 'seems', 'seemed',
              'enough',  'both',  'all',  'your' 'those',  'this',  'these',
              'their',  'the',  'that',  'some',  'our',  'neither',  'my',
              'its',  'his' 'her',  'every',  'either',  'each',  'any',  'another',
              'an',  'a',  'just',  'mere',  'such',  'merely' 'right',
              'only',  'sheer',  'even',  'especially',  'namely',  'as',  'more',
              'most',  'less', 'least',  'so',  'enough',  'too',  'pretty',  'quite',
              'rather',  'somewhat',  'sufficiently' 'same',  'different',  'such',
              'when',  'why',  'where',  'how',  'what',  'who',  'whom',  'which',
              'whether',  'why',  'whose',  'if',  'anybody',  'anyone',  'anyplace',
              'anything',  'anytime' 'anywhere',  'everybody',  'everyday',
              'everyone',  'everyplace',  'everything' 'everywhere',  'whatever',
              'whenever',  'whereever',  'whichever',  'whoever',  'whomever' 'he',
              'him',  'his',  'her',  'she',  'it',  'they',  'them',  'its',  'their', 'theirs',
              'you', 'your', 'yours', 'me', 'my', 'mine', 'i', 'we', 'us', 'much', 'and/or'
              ]


# check if a 'word' is something we need to avoid
def checkWord(check):
    for word in stop_words:
        if word == check:
            return False
    for char in punctuation_and_numbers:
        if char in check:
            return False
    return True


# read in data to dataframe
data = pd.read_csv("all-data.csv", delimiter=',', encoding="ISO-8859-1", header=None, names=["sentiment", "headlines"])

system_avg = 0
stemmer_avg = 0
baseline_avg = 0
for iteration in range(1, 101):
    # create train and test data splits
    train, test = train_test_split(data, test_size=0.20)

    # dictionary to hold words and their sentiment values
    word_sentiments = {}
    word_counts = {}
    stemmer_sentiments = {}
    stemmer_counts = {}

    # iterate through each headline and increment valid words based on attached sentiment
    for i in range(len(train)):
        sentiment = train.sentiment.iloc[i]
        headline = train.headlines.iloc[i].lower().split(' ')
        for word in headline:
            if checkWord(word):
                if word not in word_sentiments:
                    word_sentiments[word] = 1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0
                else:
                    word_sentiments[word] += 1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1

                stem = ps.stem(word)
                if stem not in stemmer_sentiments:
                    stemmer_sentiments[stem] = 1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0
                else:
                    stemmer_sentiments[stem] += 1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0
                if stem not in stemmer_counts:
                    stemmer_counts[stem] = 1
                else:
                    stemmer_counts[stem] += 1

    # normalize word sentiment values
    for word in word_sentiments.keys():
        word_sentiments[word] /= word_counts[word]
    for stem in stemmer_sentiments.keys():
        stemmer_sentiments[stem] /= stemmer_counts[stem]

    # initialize rmse variables
    system_rmse = 0
    stemmer_rmse = 0
    baseline_rmse = 0

    # iterate through test set
    for i in range(len(test)):
        true_sentiment = 1 if test.sentiment.iloc[i] == 'positive' else -1 if test.sentiment.iloc[i] == 'negative' else 0
        headline = test.headlines.iloc[i].lower().split(' ')

        sentiment_score = 0
        sentiment_abs = 0  # used to standardize score between -1 and 1
        stemmer_score = 0
        stemmer_abs = 0

        # iterate through words in headline incrementing score based on training dictionary
        for word in headline:
            if checkWord(word):
                if word in word_sentiments:
                    sentiment_score += word_sentiments[word]
                    sentiment_abs += abs(word_sentiments[word])
                stem = ps.stem(word)
                if stem in stemmer_sentiments:
                    stemmer_score += stemmer_sentiments[stem]
                    stemmer_abs += abs(stemmer_sentiments[stem])

        # standardize score
        if sentiment_abs != 0:
            sentiment_score /= sentiment_abs

        if stemmer_abs != 0:
            stemmer_score /= stemmer_abs

        system_rmse += abs(true_sentiment - sentiment_score) ** 2

        stemmer_rmse += abs(true_sentiment - stemmer_score) ** 2

        random_sentiment = random.randint(-1,1)
        baseline_rmse += abs(true_sentiment - random_sentiment) ** 2

    # calculate rmse
    normalized_system_rmse = math.sqrt(system_rmse / len(test)) / 2
    normalized_stemmer_rmse = math.sqrt(stemmer_rmse / len(test)) / 2
    normalized_baseline_rmse = math.sqrt(baseline_rmse / len(test)) / 2

    system_avg += normalized_system_rmse
    stemmer_avg += normalized_stemmer_rmse
    baseline_avg += normalized_baseline_rmse

    print("Iteration " + str(iteration))
    print("Normalized System RMSE: " + str(normalized_system_rmse))
    print("Normalized Stem RMSE: " + str(normalized_stemmer_rmse))
    print("Normalized Baseline RMSE: " + str(normalized_baseline_rmse))

print("System Average: " + str(system_avg / iteration))
print("Stem Average: " + str(stemmer_avg / iteration))
print("Baseline Average: " + str(baseline_avg / iteration))

#%%

sentence = input("Enter sentence: ")

while sentence != "quit":
    words = sentence.split(' ')

    new_score = 0
    new_score_abs = 0

    for word in words:
        if checkWord(word):
            if word in word_sentiments:
                new_score += word_sentiments[word]
                new_score_abs += abs(word_sentiments[word])

    if new_score_abs != 0:
        new_score /= new_score_abs

    print("Sentiment score: " + str(new_score))

    sentence = input("Enter sentence: ")


#%%

'''
# print out each word and its sentiment
for word in word_sentiments.keys():
    print(word + ": " + str(word_sentiments[word]))

# count the number of times a certain sentiment number appears
temp = {}
for value in word_sentiments.values():
    if value not in temp:
        temp[value] = 1
    else:
        temp[value] += 1

# print sentiment values in sorted order
keys = list(temp.keys())
keys.sort()
for key in keys:
    print(str(key) + ": " + str(temp[key]))
'''