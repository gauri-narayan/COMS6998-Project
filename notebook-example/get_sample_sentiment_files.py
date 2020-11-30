import pandas as pd
import os
import datetime
import numpy as np


def get_truncated_files(company, filename):
    last_day = "2019-01-31"
    last_datetime = datetime.datetime.strptime(last_day, '%Y-%m-%d')
    indices = np.zeros(company.shape[0])
    i = 0
    for d in company['Time']:
        date_time_obj = datetime.datetime.strptime(d[:10], '%Y-%m-%d')
        if date_time_obj <= last_datetime:
            indices[i] = 1
        else:
            indices[i] = 0
        i += 1
    company = company[indices == 1]
    del company['Unnamed: 0']
    del company['Unnamed: 0.1']
    company.to_csv("./sentiment_samples/" + filename)


# companies = [('AMZN', 'Amazon'), ('AAPL', 'Apple'), ('MSFT', 'Microsoft'),
#                  ('DIS', 'Disney'), ('GOOG', 'Google'), ('CVS', 'CVS'),
#                  ('GE', 'General Electric'), ('SAN', 'Santander'),
#                  ('GS', 'Goldman Sachs'),
#                  ('CICHY', 'China Construction Bank')]

dir = os.listdir("./sentiment/")
for filename in dir:
    print(filename)
    df = pd.read_csv("./sentiment/" + filename)
    get_truncated_files(df, filename)
