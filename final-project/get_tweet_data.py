import snscrape.modules.twitter as sntwitter
import re
import pandas as pd
import emoji
import datetime as dt
from datetime import date
from nltk.corpus import stopwords


def clean_text(text):
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


# https://github.com/JustAnotherArchivist/snscrape/issues/115
def create_corpus(company):
    name = company[0]
    keyword = company[1]
    start_date = date(2018, 12, 31)
    next_date = start_date + dt.timedelta(days=1)
    end_date = date(2020, 1, 1)
    max_tweets = 10
    tweet_content = []
    tweet_dates = []
    filename = name + "_tweets.csv"
    while next_date < end_date:
        start = start_date.strftime("%Y-%m-%d")
        next = next_date.strftime("%Y-%m-%d")
        search = keyword + ' since:' + start + ' until:' + next + ' lang:en'
        for i, t in enumerate(sntwitter.
                              TwitterSearchScraper(search).get_items()):
            if i > max_tweets:
                break
            print("collecting tweets...")
            tweet_content.append(clean_text(t.content))
            tweet_dates.append(t.date)
        start_date = next_date
        next_date = next_date + dt.timedelta(days=1)
        print("finished a batch!")
    tweet_data = {'Time': tweet_dates, 'Text': tweet_content}
    tweet_df = pd.DataFrame(tweet_data)
    tweet_df.to_csv("./tweets/" + filename)


if __name__ == "__main__":
    companies = [('AMZN', 'Amazon'), ('AAPL', 'Apple'), ('MSFT', 'Microsoft'),
                 ('DIS', 'Disney'), ('GOOG', 'Google'), ('CVS', 'CVS'),
                 ('GE', 'General Electric'), ('SAN', 'Santander'),
                 ('GS', 'Goldman Sachs'), ('CICHY', 'China Construction Bank')]
    for company in companies:
        create_corpus(company)
