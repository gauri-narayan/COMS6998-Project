from cDPM import run_cDPM
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

tweet_assignments, tweets, company = run_cDPM()

sent_df = pd.read_csv("./sentiment_Jan/" + company + "_sentiment.csv")

co_df = pd.DataFrame(columns=["date", "tweet", "topic", "sentiment"])
i = 0
dates = []
topics = []
for date in tweet_assignments:
    for assignment in tweet_assignments[date]:
        dates.append(date)
        topics.append(assignment)
        i += 1


co_df['date'] = dates
co_df['topic'] = topics
co_df['tweet'] = tweets[0:i]
sent_df = sent_df[0:i]
topic_sentiments = [0, 0]
topics_set = set(topics)

for t in topics_set:
    indices = co_df['topic'] == t
    subset_sentiments = sent_df[indices]['Sentiment']
    for s in subset_sentiments:
        s = s[1:len(s)-1]  # remove brackets
        scores = s.split(",")
        for i in range(0, 2):
            topic_sentiments[i] += float(scores[i])
    for j in range(0, 2):
        topic_sentiments[j] /= len(indices)
    co_df['sentiment'][indices] = str(topic_sentiments)

co_df.to_csv("./topic_sentiment_Jan/" + company + "_topic_sentiment.csv")
