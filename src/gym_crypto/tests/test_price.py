import unittest
import csv
import datetime
import time
import pandas as pd
import numpy as np
import os
from pandas.util.testing import assert_frame_equal
from .. import price


package_path = os.path.abspath('./')
TEST_DATA = package_path + '/gym_crypto/tests/test_data/'



class CoinTest(unittest.TestCase):
    def setUp(self):
        self.test_name = 'zilliqa'
        self.coin = price.Coin(self.test_name, test=TEST_DATA)    
    
    def test_name(self):
        self.assertEqual(self.test_name, self.coin.name)

    def test_coin_format(self):

        # Date, Open, High, Low, Close, Volume, Market Cap
        df = pd.DataFrame([
            ['Mar 30 2018', '0.5', '1.0', '0.04','0.08','100','1000']
        ])
        df = self.coin.clean_coin_data(df)
        actual = self.coin.assign_global_data(df)

        expected = pd.DataFrame([
            [1522328400.0, 0.5, 1.0, 0.04, 0.08, 100.0, 1000.0, 
                                        14652400000.0, 278462000000.0]
        ], dtype=np.float64)
        expected.columns = [
            'timestamp',
            'open', 
            'high', 
            'low', 
            'close', 
            'volume',
            'coin_marketcap',
            'global_volume',
            'global_marketcap'
        ]
        assert_frame_equal(expected, actual)

    def test_norm_columns(self):
        expected = [
            'timestamp', 
            'open', 
            'high', 
            'low',
            'close', 
            'volume', 
            'coin_marketcap',
            'global_volume', 
            'global_marketcap', 
            'norm_close', 
            'norm_volume',
            'norm_global_volume', 
            'norm_global_marketcap', 
            'norm_volume_ratio'
        ]
        self.assertEqual(set(expected), set(self.coin.data.columns))
        self.assertEqual(len(expected), len(self.coin.data.columns))

        


class GlobalTest(unittest.TestCase):
    def setUp(self):
        self.global_data = price.Global(TEST_DATA) 

    def test_global_data(self):
        self.assertTrue(self.global_data.marketcap)
        self.assertTrue(self.global_data.volume)
        self.assertTrue(self.global_data.marketcap != {})
        self.assertTrue(self.global_data.volume != {})

    def test_global_format(self):
        row = ['1522349220000','17486100000']
        target = (1522328400.0, 17486100000.0)
        self.assertEqual(self.global_data.format(row), target)


class HelperTest(unittest.TestCase):

    def test_round_timestamp(self):
        self.assertEqual(price.round_timestamp(1522349220000), 1522328400.0)

    def test_get_timestamp(self):
        self.assertEqual(price.get_timestamp('Mar 30 2018'), 1522328400.0)

    def test_valid_row(self):
        invalid_row = ['Jan','0.1','0.136','0.13','0.13','49','-']
        self.assertEqual(price.valid(invalid_row), False)
    