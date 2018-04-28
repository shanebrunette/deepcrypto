import pandas as pd
import numpy as np
from collections import namedtuple
import random
import sys

from . import price

# TODO idea: normalize on make()
# presample the data 
# sklrean.preprocessing standardscalar, minmaxscalar
# also: consider wavelet transformation on normalized data
class ActionSpace():
    def __init__(self):
        self.options = {
            'sell': 0,
            'buy': 1,
        }
        self.n = len(self.options)

class Env():
    def __init__(self, min_size=20, min_steps=10, test=False):
        self.min_steps = min_steps
        self.min_size = min_size
        self.observation_space = np.array([
            'norm_close',
            'norm_volume',
            'norm_global_volume',
            'norm_global_marketcap',
            'norm_volume_ratio',
            'balance',
            'position'
            # 'position' # open or close
        ])
        self.action_space = ActionSpace()
        self.token_position = None
        self.balance = None
        self.test = test



    def make(self, coin_list):
        coin_name = random.choice(coin_list)
        self.valid, self.data = self.get_data(coin_name)
        if not self.valid: 
            print('not valid')
            self.make(coin_list)
        else:
            print(f'environment loaded: {coin_name}')
        return self


    def get_data(self, coin_name):
        if self.test:
            coin = price.Coin(coin_name, test=self.test)
        else:
            coin = price.Coin(coin_name)
        if self.check_valid(coin):
            data = coin.data
            return True, data
        else:
            return False, pd.DataFrame(columns=coin.data.columns)


    def check_valid(self, coin):
        return coin.data.shape[0] >= self.min_size


    def reset(self, amount=1000):
        self.index = self.rand_index()
        self.balance = self.start_balance = amount
        self.token_position = None
        return self.get_state()


    def get_state(self):
        state = self.data.loc[self.index, self.observation_space]
        state['balance'] = self.balance / (self.start_balance * 2)
        state['position'] = float(1) if self.token_position else float(0)
        return np.array(state)


    def rand_index(self):
        return random.randint(1, self.data.shape[0] - self.min_steps)


    def step(self, action):
        self.index += 1
        reward = self.calc_reward(action)
        new_state = self.get_state()
        finished = self.check_finished()
        return new_state, reward, finished, {}


    def check_finished(self):
        return self.index + 2 > self.data.shape[0]


    def calc_reward(self, action):
        # TODO normalize rewards? http://karpathy.github.io/2016/05/31/rl/
        previous_balance = self.balance
        if action == 0:
            self.sell()
        elif action == 1:
            self.buy()
        return self.balance - previous_balance # TODO test


    def buy(self):
        if self.token_position:
            self.update_reward()
        else:
            self.token_position = self.open_token_position()


    def sell(self):
        if self.token_position:
            self.update_reward()
            self.token_position = None


    def update_reward(self):
        previous_price = self.get_price(self.index-1)
        current_price = self.get_price(self.index)
        self.balance += (previous_price - current_price) * self.token_position


    def open_token_position(self):
        current_price = self.get_price(self.index)
        available_funds = self.balance
        bought_amount = available_funds / current_price
        return bought_amount


    def get_price(self, index):
        return self.data.iloc[index].loc['close']
