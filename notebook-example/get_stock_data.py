from yahoo_historical import Fetcher
import pandas as pd

companies = pd.read_csv('companies-abbreviations.csv')

for co in companies['Stock Abbreviation']:
    data = Fetcher(co, [2018, 12, 31], [2020, 1, 1])
    filename = co + "_financial.csv"
    historical = data.get_historical()
    historical.to_csv("./financial/" + filename)
