import pandas as pd
import numpy as np
import re
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import datetime


def clean_text(tweet):
    text = str(tweet)
    text = ''.join([i for i in text if not i.isdigit()])  # get rid of digits
    # get rid of punctuation and symbols
    text = re.sub("[!@#$+%*:()'-]", ' ', text)
    # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    # remove URLS
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)
        (?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)
        |[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', ' ', text)
    text = text.replace('//', '')
    text = text.replace('https', '')
    text = re.sub("@[A-Za-z0-9]+", "", text)  # removing @ sign
    # removing emojis
    text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
    text = text.replace("#", "").replace("_", " ")  # removing hashtags

    en_stops = set(stopwords.words('english'))  # removing stopwords
    words = text.split()
    for word in words:
        if word in en_stops:
            words.remove(word)
    clean_text = ' '.join([str(w) for w in words])
    return clean_text.lower()


def get_date_bow(abbr, name):
    # getting vocabulary
    corpus = set()
    filename = abbr + '_tweets.csv'
    company = pd.read_csv('./tweets/' + filename)
    last_day = "2019-01-31"
    last_datetime = datetime.datetime.strptime(last_day, '%Y-%m-%d')
    indices = np.ndarray(company.shape[0])
    i = 0
    for d in company['Time']:
        date_time_obj = datetime.datetime.strptime(d[:10], '%Y-%m-%d')
        if date_time_obj <= last_datetime:
            indices[i] = True
        else:
            indices[i] = False
        i += 1

    company = company[indices == 1]

    tweet_dict = dict()
    tweets_text = list(company['Text'])
    tk = TweetTokenizer()
    for t in company['Text']:
        tweet = clean_text(t)
        tweet = tk.tokenize(tweet)
        corpus.update(tweet)

    if abbr.lower() in corpus:
        corpus.remove(abbr.lower())
    if name.lower() in corpus:
        corpus.remove(name.lower())

    # creating one-hot bag-of-words vector for each tweet
    j = 0
    dates = np.array(company['Time'])
    for tweet in company['Text']:
        vector = np.ndarray(len(corpus))
        i = 0
        for word in corpus:
            if word in tweet:
                vector[i] = 1
            else:
                vector[i] = 0
            i += 1
        tweet_date = dates[j][:10]
        if tweet_date not in tweet_dict:
            tweet_dict[tweet_date] = []
        tweet_dict[tweet_date].append(vector)
        j += 1
    return tweet_dict, tweets_text


def split_to_mult_inputs(bow_dict):
    date_trials = dict()
    for date in bow_dict:
        date_trials[date] = []
        for tweet in bow_dict[date]:
            # num_trials = np.sum(tweet)
            for i in range(0, len(tweet)):
                if tweet[i] == 1:
                    trial = np.zeros(len(tweet))
                    trial[i] = 1
                    date_trials[date].append(trial)

    return date_trials


def get_inputs():
    companies = [('AMZN', 'Amazon'), ('AAPL', 'Apple'), ('MSFT', 'Microsoft'),
                 ('DIS', 'Disney'), ('GOOG', 'Google'), ('CVS', 'CVS'),
                 ('GE', 'General Electric'), ('SAN', 'Santander'),
                 ('GS', 'Goldman Sachs'), ('CICHY', 'China Construction Bank')]
    all_companies_bow = dict()
    all_companies_mult_inputs = dict()
    all_companies_tweets = dict()
    for abbr, name in companies:
        print(abbr, name)
        bow, tweets_text = get_date_bow(abbr, name)
        all_companies_tweets[abbr] = tweets_text
        trials = split_to_mult_inputs(bow)
        all_companies_bow[abbr] = bow
        all_companies_mult_inputs[abbr] = trials
        break  # one company at a time

    return all_companies_bow, all_companies_mult_inputs, all_companies_tweets
# get_inputs()
