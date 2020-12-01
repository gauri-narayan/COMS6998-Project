import numpy as np
import torch
from torch import nn
from pyro.nn import PyroModule

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)


# split data into train and test batches
def split(data):
    cutoff = int(0.8*len(data))
    train = data[:cutoff]
    test = data[cutoff:]
    return train, test


def convert_data(df):
    y = torch.tensor(df['curr_close'].values, dtype=torch.float)
    pos_x = torch.tensor(df[['prev_close', 'pos_sentiment']].values,
                         dtype=torch.float)
    neg_x = torch.tensor(df[['prev_close', 'neg_sentiment']].values,
                         dtype=torch.float)
    return y, pos_x, neg_x


def var_model(x, y, num_iterations):
    linear_reg_model = PyroModule[nn.Linear](2, 1)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)

    def train():
        y_pred = linear_reg_model(x).squeeze(-1)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss

    for j in range(num_iterations):
        train()

    print("Learned parameters:")
    for name, param in linear_reg_model.named_parameters():
        print(name, param.data.numpy())

    w = linear_reg_model.__getattr__('weight')
    b = linear_reg_model.__getattr__('bias')
    return w, b


def mse(test, eqn, sent):
    w, b = eqn
    stock_w = w[0][0].detach().numpy()
    tweet_w = w[0][1].detach().numpy()
    bias = b[0].detach().numpy()
    obs = test['curr_close'].to_numpy()
    pred = None
    error = 0
    a = stock_w*(test['prev_close'].to_numpy())
    if sent == 'pos':
        b = tweet_w*(test['pos_sentiment'].to_numpy())
        pred = a + b + bias
    else:
        b = tweet_w*(test['neg_sentiment'].to_numpy())
        pred = a + b + bias
    error = np.square(np.subtract(obs, pred)).mean()
    return error
