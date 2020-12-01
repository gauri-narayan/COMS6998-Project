import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import json
from statistics import mean
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import datetime, timedelta
import torch
from torch import nn
from pyro.nn import PyroModule

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

plt.style.use('default')

# read in data
companies = [('AMZN', 'Amazon'), ('AAPL', 'Apple'), ('MSFT', 'Microsoft'),
             ('DIS', 'Disney'), ('GOOG', 'Google'), ('CVS', 'CVS'),
             ('GE', 'General Electric'), ('SAN', 'Santander'),
             ('GS', 'Goldman Sachs'), ('CICHY', 'China Construction Bank')]
stock_data = []
tweet_data = []
# usually n = len(companies),
# but we are testing on a sample
# (i.e., just Amazon)
n = 1
for company in companies:
    abbr = company[0]
    curr_stock = pd.read_csv('./financial/' + abbr + '_financial.csv')
    del curr_stock['Unnamed: 0']
    stock_data.append(curr_stock)
    curr_tweet = pd.read_csv('./sentiment/' + abbr + '_sentiment.csv')
    times = []
    for time in curr_tweet['Time']:
        date = time.split()[0]
        times.append(date)
    curr_tweet['Time'] = times
    tweet_data.append(curr_tweet)


def calculate_lag(stocks, tweets, lag):
    prev_stock_close = []
    curr_stock_close = []
    avg_pos = []
    avg_neg = []

    prev_time = datetime.strptime(tweets['Time'][0], '%Y-%m-%d')
    curr_time = None
    for date in stocks['Date']:
        s = datetime.strptime(date, '%Y-%m-%d').date()
        e = datetime.strptime(stocks['Date'][lag], '%Y-%m-%d').date()
        if s < e:
            continue
        index = stocks[stocks['Date'] == date].index[0]
        prev_stock_close.append(stocks['Close'][index-lag])
        curr_stock_close.append(stocks['Close'][index])
        pos = []
        neg = []
        for time in tweets['Time']:
            start = datetime.strptime(copy.copy(time), '%Y-%m-%d')
            curr_time = datetime.strptime(date, '%Y-%m-%d')
            next_day = prev_time + timedelta(days=1)
            # accounting for weekends, which 3 and 5-day lags skip over
            if lag == 1:
                b1 = (start.date() >= prev_time.date())
                b2 = (start.date() < curr_time.date())
                if b1 and b2:
                    index = tweets[tweets['Time'] == time].index[0]
                    pos.append(json.loads(tweets['Sentiment'][index])[0])
                    neg.append(json.loads(tweets['Sentiment'][index])[1])
            else:
                b1 = (start.date() >= prev_time.date())
                b2 = (start.date() < next_day.date())
                if b1 and b2:
                    index = tweets[tweets['Time'] == time].index[0]
                    pos.append(json.loads(tweets['Sentiment'][index])[0])
                    neg.append(json.loads(tweets['Sentiment'][index])[1])
        if len(pos) > 0:
            avg_pos.append(mean(pos))
        else:
            avg_pos.append(0)
        if len(neg) > 0:
            avg_neg.append(mean(neg))
        else:
            avg_neg.append(0)
        prev_time = copy.copy(curr_time)

    data = {'curr_close': curr_stock_close,
            'prev_close': prev_stock_close,
            'pos_sentiment': avg_pos,
            'neg_sentiment': avg_neg}
    col = ['curr_close', 'prev_close', 'pos_sentiment', 'neg_sentiment']
    df = pd.DataFrame(data, columns=col)
    return df


lag = 1  # sampling with just a lag of 1
lag_data = []
for i in range(n):
    df = calculate_lag(stock_data[i], tweet_data[i], lag)
    print("added data for", companies[i][1], "at lag =", lag, "...")
    lag_data.append(df)
print("finished!")


# split data into train and test batches
def split(data):
    cutoff = int(0.8*len(data))
    train = data[:cutoff]
    test = data[cutoff:]
    return train, test


train_data = []
test_data = []
for i in range(n):
    tr, ts = split(lag_data[i])
    train_data.append(tr)
    test_data.append(ts)


def var_model(x, y, num_iterations):

    # Regression model
    linear_reg_model = PyroModule[nn.Linear](2, 1)

    # Define loss and optimize
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
    # num_iterations = 1500

    def train():
        # run the model forward on the data
        y_pred = linear_reg_model(x).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        return loss

    for j in range(num_iterations):
        loss = train()
        if (j + 1) % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))

    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in linear_reg_model.named_parameters():
        print(name, param.data.numpy())

    weights = linear_reg_model.__getattr__('weight')
    bias = linear_reg_model.__getattr__('bias')
    return weights, bias


def convert_data(df):
    y = torch.tensor(df['curr_close'].values,
                     dtype=torch.float)
    pos_x = torch.tensor(df[['prev_close', 'pos_sentiment']].values,
                         dtype=torch.float)
    neg_x = torch.tensor(df[['prev_close', 'neg_sentiment']].values,
                         dtype=torch.float)
    return y, pos_x, neg_x


def granger_causality(i, sent):
    lag_one = calculate_lag(stock_data[i], tweet_data[i], 1)
    if sent == 'pos':
        grangercausalitytests(lag_one[['prev_close', 'pos_sentiment']],
                              maxlag=1, verbose=True)
    else:
        grangercausalitytests(lag_one[['prev_close', 'neg_sentiment']],
                              maxlag=1, verbose=True)


# inference
iterations = 1500
pos_lr = []
neg_lr = []
for i in range(n):
    y, pos_x, neg_x = convert_data(train_data[i])
    print("\n", companies[i][1], "with positive sentiment: ")
    pos_equation = var_model(pos_x, y, iterations)
    pos_lr.append(pos_equation)
    granger_causality(i, 'pos')
    print("\n", companies[i][1], "with negative sentiment: ")
    neg_equation = var_model(neg_x, y, iterations)
    neg_lr.append(neg_equation)
    granger_causality(i, 'neg')


# evaluation
def mse(test, eqn, sent):
    w, b = eqn
    stock_w = w[0][0].detach().numpy()
    tweet_w = w[0][1].detach().numpy()
    bias = b[0].detach().numpy()
    obs = test['curr_close'].to_numpy()
    pred = None
    if sent == 'pos':
        a = stock_w*(test['prev_close'].to_numpy())
        b = tweet_w*(test['pos_sentiment'].to_numpy())
        pred = a + b + bias
    else:
        a = stock_w*(test['prev_close'].to_numpy())
        b = tweet_w*(test['neg_sentiment'].to_numpy())
        pred = a + b + bias
    error = np.square(np.subtract(obs, pred)).mean()
    return error


for i in range(n):
    error = mse(test_data[i], pos_lr[i], 'pos')
    print('The MSE of', companies[i][1], 'is',
          error, 'when using positive sentiment as a parameter.')
    error = mse(test_data[i], pos_lr[i], 'neg')
    print('The MSE of', companies[i][1], 'is',
          error, 'when using negative sentiment as a parameter.')

# comparing trend of stock to trend of
# positive and negative sentiment over the year
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(stock_data[0].iloc[1:]['Date'],
         lag_data[0]['prev_close'], color='b')
ax1.plot(stock_data[0].iloc[1:]['Date'],
         lag_data[0]['pos_sentiment'], color='g')
ax1.set(xlabel='Date',
        ylabel='Closing Price',
        title='Amazon Stock and Average Positive Sentiment')
ax2.plot(stock_data[0]['Date'],
         stock_data[0]['Close'], color='b')
ax2.plot(stock_data[0].iloc[1:]['Date'],
         lag_data[0]['neg_sentiment'], color='r')
ax2.set(xlabel='Date',
        title='Amazon Stock and Average Negative Sentiment')
fig.savefig("Amazon_VAR_Stock_Prediction.png")
