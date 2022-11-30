import pandas as pd
import math

# avoid punctuation and numbers
punctuation_and_numbers = ["!", "@", "#", "$", "%", "'", "^", "&", "*", "(", ")", "{", "}", "[", "]", "\\", "|", "=",
                           "+", "/", "?", "-", "_", ".", "<", ">", "`", "~", ";", ":", ",",
                           '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# words to skip over
stop_words = ['a', 'the', 'an', 'and', 'or', 'but', 'about', 'above', 'after', 'along', 'amid', 'among',
              'as', 'at', 'by', 'for', 'from', 'in', 'into', 'like', 'near', 'of', 'off', 'on',
              'onto', 'out', 'over', 'past', 'per', 'plus', 'since', 'till', 'to', 'under', 'until', 'up',
              'via', 'vs', 'with', 'that', 'could', 'may', 'might', 'must',
              'need', 'ought', 'shall', 'should', 'will', 'would', 'have', 'had', 'has', 'having', 'be',
              'is', 'am', 'are', 'was', 'were', 'being', 'been', 'get', 'gets', 'got', 'gotten',
              'getting', 'seem', 'seeming', 'seems', 'seemed', 'did', 'do', 'does',
              'enough',  'both',  'all',  'your' 'those',  'this',  'these',
              'their',  'the',  'that',  'some',  'our',  'my',
              'its',  'his' 'her',  'every',  'either',  'each',  'any',  'another',
              'an',  'a',  'just',  'mere',  'such',  'merely' 'right',
              'only',  'sheer',  'even',  'especially',  'namely',  'as',  'more',
              'most', 'least',  'so',  'enough',  'too',  'pretty',  'quite',
              'somewhat',  'sufficiently' 'same',  'different',  'such',
              'when',  'why',  'where',  'how',  'what',  'who',  'whom',  'which',
              'whether',  'why',  'whose',  'if',  'anybody',  'anyone',  'anyplace',
              'anything',  'anytime' 'anywhere',  'everybody',  'everyday',
              'everyone',  'everyplace',  'everything' 'everywhere',  'whatever',
              'whenever',  'wherever',  'whichever',  'whoever',  'whomever' 'he',
              'him',  'his',  'her',  'she',  'it',  'they',  'them',  'its',  'their', 'theirs',
              'you', 'your', 'yours', 'me', 'my', 'mine', 'i', 'we', 'us', 'much', 'and/or', 'wo', 'ca', 'mus', 'sha'
              ]

# words that reverse the sentiment of subsequent words
flip_words = ['no', 'not', 'rather', 'never', 'none', 'nobody', 'nothing',
              'neither', 'nor', 'nowhere', 'cannot', 'without', 'n\'t']

# words that reduce the sentiment of subsequent words
diminish_words = ['hardly', 'less', 'little', 'rarely', 'scarcely', 'seldom']
dimVal = 0.5


# check if a word should be skipped
def checkWord(check):
    for word in stop_words:
        if word == check:
            return False
    for char in punctuation_and_numbers:
        if check != 'n\'t' and char in check:
            return False
    return True


# check if a word flips sentiment for subsequent words
def checkFlip(check):
    for word in flip_words:
        if word == 'n\'t' and word in check:
            return True
        elif word == check:
            return True
    return False


# check if a word diminishes the sentiment for subsequent words
def checkDiminish(check):
    for word in diminish_words:
        if word == check:
            return True
    return False


# assigns sentiment values to words
def assignSentiment(flipNum=0, diminish=0.5, train=pd.DataFrame):
    # initialize variables
    word_sentiments = {}
    word_counts = {}
    flip = 0
    increment = 1

    # iterate through each headline and increment valid words based on attached sentiment
    for i in range(len(train)):
        sentiment = train.sentiment.iloc[i]
        headline = train.headlines.iloc[i].lower().split(' ')
        for word in headline:
            if checkWord(word):  # check to skip word or not
                if checkFlip(word):  # check if increment should be flipped for following words
                    flip = flipNum  # the next flipNum words will have reversed increments
                    continue
                if checkDiminish(word):  # check if increment should be reduced for the following word
                    increment = diminish  # the next increment will be diminished
                    continue
                if flip == 0:  # normal sentiment increment
                    if word not in word_sentiments:
                        word_sentiments[word] = increment if sentiment == 'positive' else increment * -1 if sentiment == 'negative' else 0
                    else:
                        word_sentiments[word] += increment if sentiment == 'positive' else increment * -1 if sentiment == 'negative' else 0
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
                else:  # flipped sentiment increment
                    if word not in word_sentiments:
                        word_sentiments[word] = increment if sentiment == 'negative' else increment * -1 if sentiment == 'positive' else 0
                    else:
                        word_sentiments[word] += increment if sentiment == 'negative' else increment * -1 if sentiment == 'positive' else 0
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
                    flip -= 1
                if increment != 1:  # return increment to normal
                    increment = 1
    # normalize values
    for word in word_sentiments.keys():
        word_sentiments[word] /= word_counts[word]
    return word_sentiments


def calculateRMSE(word_sentiments, validation=pd.DataFrame):
    rmse = 0

    # iterate through validation set
    for i in range(len(validation)):
        true_sentiment = 1 if validation.sentiment.iloc[i] == 'positive' else -1 if validation.sentiment.iloc[i] == 'negative' else 0
        headline = validation.headlines.iloc[i].lower().split(' ')

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

        # increment rmse
        rmse += abs(true_sentiment - sentiment_score) ** 2

    # calculate and return rmse score
    return math.sqrt(rmse / len(validation)) / 2
