import pandas as pd
import numpy as np
import nltk
import re
import emoji
import math
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import datetime
#from collections import OrderedDict
#from collections import OrderedDict

def clean_text(tweet):
    text = str(tweet)
    text = ''.join([i for i in text if not i.isdigit()]) # get rid of digits
    text = re.sub("[!@#$+%*:()'-]", ' ', text) # get rid of punctuation and symbols
    # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', ' ', text) # remove URLS
    text = text.replace('//', '')
    text = text.replace('https', '')
    text = re.sub("@[A-Za-z0-9]+","",text) # removing @ sign
    text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI) # removing emojis
    text = text.replace("#", "").replace("_", " ") # removing hastags

    en_stops = set(stopwords.words('english')) # removing stopwords
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
        date_time_obj = datetime.datetime.strptime(d[:10],'%Y-%m-%d')
        #print(date_time_obj.date())
        if date_time_obj<=last_datetime:
            indices[i]=True
        else:
            indices[i]=False
        i+=1

    company = company[indices==1]

    tweet_dict = dict()
    tweets_text = list(company['Text'])
    tk = TweetTokenizer()
    for t in company['Text']:
        tweet = clean_text(t)
        tweet = tk.tokenize(tweet)
        #print(tweet)
        corpus.update(tweet)
    #print(len(corpus))
    # removing company name from corpus
    if abbr.lower() in corpus:
        corpus.remove(abbr.lower())
    if name.lower() in corpus:
        corpus.remove(name.lower())
    #creating one-hot bag-of-words vector for each tweet
    input = []
    j=0
    dates = np.array(company['Time'])
    for tweet in company['Text']:
        vector = np.ndarray(len(corpus))
        i =0
        for word in corpus:
            if word in tweet:
                vector[i]=1
            else:
                vector[i]=0
            i+=1
        tweet_date = dates[j][:10]
        #print(tweet_date)
        if tweet_date not in tweet_dict:
            tweet_dict[tweet_date] = []
        tweet_dict[tweet_date].append(vector)
        #tweets[tweet_date].append(tweet)
        #print(tweet_dict)
        j+=1
    #print(tweet_dict)
    return tweet_dict, tweets_text


def split_to_mult_inputs(bow_dict):
    date_trials = dict()
    for date in bow_dict:
        #print(date)
        date_trials[date] = []
        for tweet in bow_dict[date]:
            num_trials = np.sum(tweet)
            #print(num_trials)
            for i in range(0, len(tweet)):
                if tweet[i]==1:
                    trial = np.zeros(len(tweet))
                    trial[i]=1
                    date_trials[date].append(trial)

    return date_trials


def get_inputs():
    #companies = pd.read_csv('COMS6998-Project/companies-abbreviations.csv')
    # companies = [('AMZN', 'Amazon'), ('AAPL', 'Apple'), ('MSFT', 'Microsoft'),
    #              ('DIS', 'Disney'), ('GOOG', 'Google'), ('CVS', 'CVS'),
    #              ('GE', 'General Electric'), ('SAN', 'Santander'),
    #              ('GS', 'Goldman Sachs'), ('CICHY', 'China Construction Bank')]
    #companies = pd.read_csv('COMS6998-Project/companies-abbreviations.csv')
    companies = [('GS', 'Goldman Sachs'), ('CICHY', 'China Construction Bank')]
    all_companies_bow = dict()
    all_companies_mult_inputs = dict()
    #print(companies[9])
    all_companies_tweets = dict()
    for abbr, name in companies:
        print(abbr, name)
        bow, tweets_text = get_date_bow(abbr,name)
        all_companies_tweets[abbr] = tweets_text
        #print(bow['2018-12-31'])
        #all_companies_bow[abbr] = bow
        trials = split_to_mult_inputs(bow)
        all_companies_bow[abbr] = bow
        #print(all_companies_bow)
        #print(bow)
        all_companies_mult_inputs[abbr] = trials
        #print(trials['2018-12-31'])
        #for trial in trials['2018-12-31']:
            #print(sum(trial))
        break #one company at a time


    return all_companies_bow, all_companies_mult_inputs, all_companies_tweets

#c_bow, c_trials = get_inputs()