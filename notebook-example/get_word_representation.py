import pandas as pd
import numpy as np


def input(file):
    # getting vocabulary
    corpus = set()
    filename = file + '_tweets.csv'
    company = pd.read_csv('./tweets/' + filename)
    for t in company['Text']:
        tweet = set(t.split())
        corpus.update(tweet)

    # removing company name from corpus
    name = file.split()
    for n in name:
        corpus.remove(n.lower())

    # creating one-hot bag-of-words vector for each tweet
    input = []
    for tweet in company['Text']:
        vector = []
        for word in tweet:
            if word in corpus:
                vector.append(1)
            else:
                vector.append(0)
        input.append(np.array(vector))

    return np.array(input)


companies = pd.read_csv('./companies-abbreviations.csv')
sample_batch = 23
for file in companies['Company'][:sample_batch]:
    input(file)
