# processes raw price data

import datetime
import time
import pandas as pd
import numpy as np
import csv
import os
from . import norm


data_path = os.path.abspath('../../data/')
COIN_DIR = data_path+'/raw/'
GLOBAL_DIR = data_path+'/global_raw/'
del data_path


class Coin:
    def __init__(self, 
                 coin, 
                 coin_dir=COIN_DIR, 
                 global_dir=GLOBAL_DIR,
                 test=False):
        
        self.name = coin
        if test:
            self.coin_dir = self.global_dir = test
        else:
            self.coin_dir = coin_dir
            self.global_dir = global_dir
        self.data = self.get_data(coin)

    def get_data(self, coin):
        df = self.get_coin_data(coin)
        df = self.assign_global_data(df)
        df = norm.normalize(df) 
        return df

    def get_coin_data(self, coin):
        df = pd.read_csv(self.coin_dir + coin +'.csv')
        df = self.clean_coin_data(df)
        return df

    def clean_coin_data(self, df):
        df = df.replace('-', np.NaN)
        df = df.dropna()
        df = df.tail(180) # TODO how many days
        df.columns = [
            'timestamp',
            'open', 
            'high', 
            'low', 
            'close', 
            'volume',
            'coin_marketcap'
        ]
        df['timestamp'] = df['timestamp'].apply(lambda x: get_timestamp(x))
        df = df.astype(np.float64)
        df = df.reindex(index=df.index[::-1])
        return df

    def assign_global_data(self, df):
        global_data = Global(self.global_dir)
        df = df.assign(global_volume=df['timestamp'].apply(
                    lambda time: global_data.volume[time]))
        df = df.assign(global_marketcap=df['timestamp'].apply(
                    lambda time: global_data.marketcap[time]))
        return df


class Global:
    def __init__(self, global_dir=GLOBAL_DIR):
        self.global_dir = global_dir
        self.marketcap = self.get_data('total_marketcap')
        self.volume = self.get_data('total_volume')

    def get_data(self, file_name):
        clean_data = {}
        with open(self.global_dir + file_name + '.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                if valid(row):
                    timestamp, value = self.format(row)
                    clean_data[timestamp] = value
        return clean_data

    def format(self, row):
        return float(round_timestamp(row[0])), float(row[1])


def round_timestamp(timestamp_str):
    dt = datetime.datetime.fromtimestamp(float(timestamp_str) / 1e3)
    dt = datetime.datetime(*dt.timetuple()[:3])
    return time.mktime(dt.timetuple())


def get_timestamp(date_str):
    dt = datetime.datetime.strptime(date_str, '%b %d %Y')
    return time.mktime(dt.timetuple())


def valid(row):
    return len([cell for cell in row if cell != '-']) == len(row)


if __name__ == '__main__':
    Coin('zilliqa')



