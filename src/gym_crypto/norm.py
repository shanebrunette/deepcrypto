# converts coin price into normalized data for env state output

import math
import pandas as pd
import numpy as np
import math
import sys


def normalize(df):
    norm_df = norm_columns(df)
    df = merge(df, norm_df)
    df = df.dropna()
    return df


def norm_columns(df):
    norm_df = init_norm_df(df)
    norm_df = get_prev_current_diff(df, norm_df)
    norm_df = get_volume_ratio(df, norm_df)
    return norm_df


def init_norm_df(df):
    norm_df = pd.DataFrame(dtype=np.float64)
    norm_df['timestamp'] = df['timestamp'][1:]
    return norm_df


def get_prev_current_diff(df, norm_df):
    price_diff_cols = [
        'close',
        'volume',
        'global_volume',
        'global_marketcap'
    ]
    for col_name in price_diff_cols:
        norm_df[f'norm_{col_name}'] = log_diff(df[col_name])
    return norm_df


def get_volume_ratio(df, norm_df):
    norm_df['norm_volume_ratio'] = df['volume'][1:]/df['coin_marketcap'][1:]
    return norm_df
    

def log_diff(col):
    # np.log(series.loc[col]).diff(1)) TODO
    log_col = col.apply(lambda x: math.log(x) if x > 0.0 else 1e-15)
    return log_col - log_col.shift(-1) 


def volatility(df):
    return (df['close'] - df['low'])/(df['high']-df['low'])


def merge(df, norm_df):
    return df.merge(norm_df, left_on='timestamp', right_on='timestamp', how='left')



