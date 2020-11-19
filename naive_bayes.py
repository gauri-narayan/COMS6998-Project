import pandas as pd
import re
import emoji
import math
from nltk.corpus import stopwords

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
train = pd.read_csv('tweet-sentiment-extraction/train.csv')
train_tweets = []
for tweet in train['selected_text']:
    train_tweets.append(clean_text(tweet))
train_sentiment = train['sentiment']

# computing P(pos), P(neg), and P(neutral)
p_pos = 0
p_neg = 0
p_neutral = 0
for sentiment in train_sentiment:
    if sentiment == 'positive':
        p_pos += 1
    elif sentiment == 'negative':
        p_neg += 1
    else:
        p_neutral += 1
# print(p_pos)
# p_labels
p_pos = p_pos/len(train_sentiment)
p_neg = p_neg/len(train_sentiment)
p_neutral = p_neg/len(train_sentiment)

# computing P(word)
vocabulary = get_word_probabilities(train_tweets)
# vocabulary = dict()
# for tweet in train_tweets:
#     text = str(tweet).split()
#     for word in text:
#         if word in vocabulary:
#             vocabulary[word] += 1
#         else:
#             vocabulary[word] = 1
# vocabulary_size = sum(vocabulary.values())
# vocabulary = {k: v/vocabulary_size for k, v in vocabulary.items()}

# computing conditional P(word|label)
positive = get_sentiment_groups('positive', train_sentiment, train_tweets)
negative = get_sentiment_groups('negative', train_sentiment, train_tweets)
neutral = get_sentiment_groups('neutral', train_sentiment, train_tweets)
positive_vocab = get_word_probabilities(positive)
negative_vocab = get_word_probabilities(negative)
neutral_vocab = get_word_probabilities(neutral)

companies = pd.read_csv('COMS6998-Project/companies-abbreviations.csv')
sample_batch = 23
for file in companies['Company'][:sample_batch]:
    filename = file + '_tweets.csv'
    company = pd.read_csv('COMS6998-Project/sample_tweets/' + filename)
    sentiments = []
    for tweet in company['Text']:
        p_words_pos = 0
        p_words_neg = 0
        p_words_neutral = 0
        p_words = 0
        for word in tweet.split():
            if word in vocabulary:
                p_words += math.log(vocabulary[word])
            if word in positive_vocab:
                p_words_pos += math.log(positive_vocab[word])
            if word in negative_vocab:
                p_words_neg += math.log(negative_vocab[word])
            if word in neutral_vocab:
                p_words_neutral += math.log(neutral_vocab[word])
        pos_likelihood = (p_words_pos - p_words) + math.log(p_pos)
        neg_likelihood = (p_words_neg - p_words) + math.log(p_neg)
        neutral_likelihood = (p_words_neutral - p_words) + math.log(p_neutral)
        prediction = [pos_likelihood, neg_likelihood, neutral_likelihood]
        # print(prediction)
        sentiments.append(prediction)
    company['Sentiment'] = sentiments
    company.to_csv('./sentiments/' + file + '_tweet_sentiment.csv')

'''
for each company csv:
    sentiments = []
    read in tweets
    for each tweet:
        break down tweet into list of words
        must calculate P(label|words) = P(label) [P(word|label)/P(word) for each word]
        we have P(label) already
        words_label = 0
        words = 0
        for each word in tweet:
            words_label = calculate P(word|label) and multiply across all words
            words = calculate P(word) and multiply across all words
        alpha = words_label/words
        pos_likelihood = p_pos * alpha
        neg_likelihood = p_neg * alpha
        neutral_likelihood = p_neutral * alpha
        prediction = [log(pos_likelihood), log(neg_likelhihood), log(neutral_likelihood)]
        sentiment.append(prediction)
    make new dataframe with tweets in one column and sentiments in the other
    save new dataframe as company_title_sentiment.csv
'''



# https://www.dataquest.io/blog/naive-bayes-tutorial/
'''
use log probs!!!

P(label): compute probability of sentiment label --> # of pos tweets/total tweets
P(word|label): compute probability of each word given label --> total occurrence of the word in that label / total words in that particular label
P(word): probability of each word --> # of instances of particular word/vocab (?) multiplied across all words

for 'prediction', have to compute probability of all labels given tweet,
i.e. compute tweet|pos, tweet|neg, and tweet|neutral
greatest probability is label
if unknown word --> 1/vocabulary_size
'''
