import pandas as pd
import re
import emoji
import math
from nltk.corpus import stopwords


def clean_text(tweet):
    text = str(tweet)
    text = ''.join([i for i in text if not i.isdigit()])  # get rid of digits
    text = re.sub("[!@#$+%*:()'-]",
                  ' ', text)  # get rid of punctuation and symbols
    # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    text = re.sub(r'''(?i)\b((?:https?://|
                  www\d{0,3}[.]|[a-z0-9.\-]+
                  [.][a-z]{2,4}/)(?:[^\s()<>]+
                  |\(([^\s()<>]+|(\([^\s()<>]+
                  \)))*\))+(?:\(([^\s()<>]+
                  |(\([^\s()<>]+\)))*\)|
                  [^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
                  ' ', text)  # remove URLS
    text = text.replace('//', '')
    text = text.replace('https', '')
    text = re.sub("@[A-Za-z0-9]+", "", text)  # removing @ sign
    text = ''.join(c for c in text if
                   c not in emoji.UNICODE_EMOJI)  # removing emojis
    text = text.replace("#", "").replace("_", " ")  # removing hastags

    en_stops = set(stopwords.words('english'))  # removing stopwords
    words = text.split()
    for word in words:
        if word in en_stops:
            words.remove(word)
    clean_text = ' '.join([str(w) for w in words])
    return clean_text.lower()


def get_sentiment_groups(label, sentiment, tweets):
    group = []
    i = 0
    for s in sentiment:
        if s == label:
            group.append(tweets[i])
        i += 1
    return group


def get_word_probabilities(tweets):
    probabilities = dict()
    for tweet in tweets:
        words = str(tweet).split()
        for word in words:
            if word in probabilities:
                probabilities[word] += 1
            else:
                probabilities[word] = 1
    size = sum(probabilities.values())
    probabilities = {k: v/size for k, v in probabilities.items()}
    return probabilities


# training stage
train = pd.read_csv('./tweet-sentiment-extraction/train.csv')
train_tweets = []
for tweet in train['selected_text']:
    train_tweets.append(clean_text(tweet))
train_sentiment = train['sentiment']

# computing P(pos) and P(neg)
p_pos = 0
p_neg = 0
for sentiment in train_sentiment:
    if sentiment == 'positive':
        p_pos += 1
    else:
        p_neg += 1
p_pos = p_pos/len(train_sentiment)
p_neg = p_neg/len(train_sentiment)

# computing P(word)
vocabulary = get_word_probabilities(train_tweets)

# computing conditional P(word|label)
positive = get_sentiment_groups('positive', train_sentiment, train_tweets)
negative = get_sentiment_groups('negative', train_sentiment, train_tweets)
positive_vocab = get_word_probabilities(positive)
negative_vocab = get_word_probabilities(negative)

companies = [('AMZN', 'Amazon'), ('AAPL', 'Apple'), ('MSFT', 'Microsoft'),
             ('DIS', 'Disney'), ('GOOG', 'Google'), ('CVS', 'CVS'),
             ('GE', 'General Electric'), ('SAN', 'Santander'),
             ('GS', 'Goldman Sachs'), ('CICHY', 'China Construction Bank')]
for company in companies:
    abbr = company[0]
    filename = abbr + '_tweets.csv'
    company = pd.read_csv('./tweets/' + filename)
    sentiments = []
    for tweet in company['Text']:
        p_words_pos = 0
        p_words_neg = 0
        p_words = 0
        for word in tweet.split():
            if word in vocabulary:
                p_words += math.log(vocabulary[word])
            if word in positive_vocab:
                p_words_pos += math.log(positive_vocab[word])
            if word in negative_vocab:
                p_words_neg += math.log(negative_vocab[word])
        pos_likelihood = (p_words_pos - p_words) + math.log(p_pos)
        neg_likelihood = (p_words_neg - p_words) + math.log(p_neg)
        prediction = [pos_likelihood, neg_likelihood]
        sentiments.append(prediction)
    company['Sentiment'] = sentiments
    company.to_csv('./sentiment/' + abbr + '_sentiment.csv')
