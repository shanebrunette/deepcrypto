import unittest
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import math
from .. import norm
from .. import price


class NormTest(unittest.TestCase):
    def setUp(self):
        self.coin_name = 'bitcoin'
        self.coin = price.Coin(self.coin_name)

    
    def slope(self):
        coin = self.coin.data
        data = [
            go.Scatter(
                x=coin['timestamp'], # assign x as the dataframe column 'x'
                y=coin['norm_close'],
                name='close'
            ),
            go.Scatter(
                x=coin['timestamp'], # assign x as the dataframe column 'x'
                y=coin['norm_volume'],
                name='volume'
            ),
            go.Scatter(
                x=coin['timestamp'], # assign x as the dataframe column 'x'
                y=coin['norm_global_volume'],
                name='global vol'
            ),
            go.Scatter(
                x=coin['timestamp'], # assign x as the dataframe column 'x'
                y=coin['norm_global_marketcap'],
                name='global marketcap'
            ),
            go.Scatter(
                x=coin['timestamp'], # assign x as the dataframe column 'x'
                y=coin['norm_volume_ratio'],
                name='volume ratio'
            )
        ]

        url = py.plot(data, filename=self.coin_name)

    def correlations(self):
        print(self.coin.data.tail(10))