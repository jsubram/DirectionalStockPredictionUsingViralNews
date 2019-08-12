from src.utils import Utils
import pandas as pd
import numpy as np


class StockProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.amazon_stock_prices = None
        self.apple_stock_prices = None

    def getDatafrane(self, filename):
        return pd.read_csv(self.base_path + '/CHARTS/' + filename, header=None)

    def formatData(self):
        cols = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Other']
        self.amazon_stock_prices.columns = cols
        self.apple_stock_prices.columns = cols
        self.amazon_stock_prices['Date'] = pd.to_datetime(self.amazon_stock_prices['Date'])
        self.apple_stock_prices['Date'] = pd.to_datetime(self.apple_stock_prices['Date'])
        # Filtering out date from 2018 onwards
        self.amazon_stock_prices = self.amazon_stock_prices[self.amazon_stock_prices['Date'] >= '2018-1-1 00:00:00']
        self.apple_stock_prices = self.apple_stock_prices[self.apple_stock_prices['Date'] >= '2018-1-1 00:00:00']

        self.amazon_stock_prices['Date'] = self.amazon_stock_prices['Date'].dt.date
        self.apple_stock_prices['Date'] = self.apple_stock_prices['Date'].dt.date

        # converting GMT to EST
        self.amazon_stock_prices['Time'] = pd.to_datetime(self.amazon_stock_prices['Time'])
        self.amazon_stock_prices['Time'] = self.amazon_stock_prices['Time'] - pd.Timedelta('04:00:00')
        self.amazon_stock_prices['Time'] = self.amazon_stock_prices['Time'].dt.time

        self.apple_stock_prices['Time'] = pd.to_datetime(self.apple_stock_prices['Time'])
        self.apple_stock_prices['Time'] = self.apple_stock_prices['Time'] - pd.Timedelta('04:00:00')
        self.apple_stock_prices['Time'] = self.apple_stock_prices['Time'].dt.time

    def findStockDirection(self):
        """ To check if stock is moving up or down
           if open < close => UP => +1
           if open > Close => DOWN => -1
           """
        self.amazon_stock_prices['Direction'] = np.where(
            self.amazon_stock_prices['Open'] < self.amazon_stock_prices['Close'], 1, -1)
        self.apple_stock_prices['Direction'] = np.where(
            self.apple_stock_prices['Open'] < self.apple_stock_prices['Close'], 1, -1)

    def loadDataForInterval(self, time_interval):
        """
        Loading data for a particular interval
        :param time_interval: prediction timeframe
        :return: Stock charts for amazon and apple for the particular timeframe
        """
        self.amazon_stock_prices = self.getDatafrane('AMAZON' + time_interval + '.csv')
        self.apple_stock_prices = self.getDatafrane('APPLE' + time_interval + '.csv')
        self.formatData()
        self.findStockDirection()
        return self.amazon_stock_prices, self.apple_stock_prices
