import pickle
import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib import style
import Tkinter

class Program:
    def __init__(self):
        self.ticker_symbol = 'GOOGL'
        self.df = None

    def download(self):
        self.df = quandl.get("WIKI/" + self.ticker_symbol)
        with open('temp_stock_data.pickle', 'wb') as f:
            pickle.dump(self.df, f)

    def load(self):
        pickle_in = open('temp_stock_data.pickle', 'rb')
        self.df = pickle.load(pickle_in)

    def format_data(self):

        self.df = self.df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
        self.df['HL_PCT'] = (self.df['Adj. High'] - self.df['Adj. Low']) / self.df['Adj. Close'] * 100.0
        self.df['PCT_change'] = (self.df['Adj. Close'] - self.df['Adj. Open']) / self.df['Adj. Open'] * 100.0
        self.df = self.df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

        print(self.df.head())

        self.df = self.df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
        forecast_col = 'Adj. Close'
        self.df.fillna(value=-99999, inplace=True)
        forecast_out = int(math.ceil(0.01 * len(self.df)))
        self.df['label'] = self.df[forecast_col].shift(-forecast_out)

        self.X = np.array(self.df.drop(['label'], 1))

        # Scale all of our features to be between -1 and 1. This is a standard in machine learning
        self.X = preprocessing.scale(self.X)

        # Split off most recent data
        self.X_lately = self.X[-forecast_out:]
        self.X = self.X[:-forecast_out]

        self.df.dropna(inplace=True)
        self.y = np.array(self.df['label'])

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X, self.y, test_size=0.2)

    def create_classifier(self,k):
        self.clf = svm.SVR(kernel=k)
        #self.clf = LinearRegression(n_jobs=-1)
        self.clf.fit(self.X_train, self.y_train)
        confidence = self.clf.score(self.X_test, self.y_test)
        print confidence
        return confidence

    def forecast(self):
        self.forecast_set = self.clf.predict(self.X_lately)
        self.df['Forecast'] = np.nan

    def plot(self):
        style.use('ggplot')
        self.df['Forecast'] = np.nan

        last_date = self.df.iloc[-1].name
        last_unix = time.mktime(last_date.timetuple())
        one_day = 86400
        next_unix = last_unix + one_day

        for i in self.forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += 86400
            self.df.loc[next_date] = [np.nan for _ in range(len(self.df.columns) - 1)] + [i]

        self.df['Adj. Close'].plot()
        self.df['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()


def main():
    p = Program()

    p.download()
    p.load()
    p.format_data()
    p.train()
    p.create_classifier('linear')
    p.forecast()
    p.plot()

if __name__ == '__main__':
    main()
