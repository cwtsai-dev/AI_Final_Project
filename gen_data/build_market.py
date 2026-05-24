"""Build all preprocessed data for one market.

Run from the gen_data/ directory (paths are relative to ../dataset):
    python build_us_market.py nd100     # US:    nd100 | sp500
    python build_us_market.py zz500     # China: hs300 | zz500

Pipeline:
  1. industry graph:
       US markets    -> dataset/{market}/industry.npy   (same-GICS-sector = 1, diag = 0,
                        from dataset/us_sectors.csv)
       China markets -> skipped; train_predict_data slices dataset/A_stock_industry_matrx.csv
  2. corr matrices -> dataset/corr/{market}/*.csv      (monthly Pearson, via generate_relation)
  3. .pkl samples  -> dataset/data_train_predict_{market}/1_hy/*.pkl  (via train_predict_data)
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_relation import generate_corr_matrices
import train_predict_data as tpd

DATASET = '../dataset'


def build_industry(market):
    codes = sorted(pd.read_csv(f'{DATASET}/{market}_org.csv', usecols=['kdcode'])['kdcode'].unique().tolist())
    sec = pd.read_csv(f'{DATASET}/us_sectors.csv').drop_duplicates(subset='ticker')
    sector = dict(zip(sec['ticker'], sec['sector']))
    missing = [c for c in codes if c not in sector]
    assert not missing, f"[{market}] tickers without a sector label: {missing}"

    labels = [sector[c] for c in codes]
    N = len(codes)
    mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j and labels[i] == labels[j]:
                mat[i, j] = 1.0

    os.makedirs(f'{DATASET}/{market}', exist_ok=True)
    np.save(f'{DATASET}/{market}/industry.npy', mat)
    print(f"[{market}] industry.npy: {mat.shape}, same-sector edges={int(mat.sum())}")
    return codes


def build_pkls(market):
    df_row = pd.read_csv(f'{DATASET}/{market}_org.csv')
    df = tpd.get_label(df=df_row, horizon=1)
    df = tpd.cal_rolling_mean_std(df, cal_cols=['close', 'volume'], lookback=5)
    df = tpd.group_and_norm(df, base_cols=['close_mean', 'close_std', 'volume_mean', 'volume_std'], n=4)
    df_all = df.copy()
    df = df[(df['dt'] >= '2018-01-01') & (df['dt'] <= '2024-12-31')]
    stock_trade_dt_s_all = sorted(df_all['dt'].unique().tolist())
    stock_trade_dt_s = sorted(df['dt'].unique().tolist())
    filter_code_s = tpd.filter_code(df)
    print(f"[{market}] building {len(stock_trade_dt_s)} daily samples for {len(filter_code_s)} stocks")
    for dt in tqdm(stock_trade_dt_s):
        relation_dt = tpd.get_relation_dt(str_year=dt[:4], str_month=dt[5:7], stock_trade_dt_s=stock_trade_dt_s)
        tpd.generate_train_predict_data_by_date(
            dt=dt, df=df_all, relation_dt=relation_dt,
            stock_trade_dt_s_all=stock_trade_dt_s_all, filter_code_s=filter_code_s,
            market=market, horizon=1, relation_type='hy', lookback=20, threshold=0.2, norm=True)


US_MARKETS = ('nd100', 'sp500')
CN_MARKETS = ('hs300', 'zz500')

if __name__ == '__main__':
    market = sys.argv[1]
    assert market in US_MARKETS + CN_MARKETS, f"unknown market: {market}"
    if market in US_MARKETS:
        print(f"=== Step 1/3: industry.npy ({market}) ===")
        build_industry(market)
    else:
        print(f"=== Step 1/3: industry skipped ({market} slices A_stock_industry_matrx.csv) ===")
    print(f"=== Step 2/3: correlation matrices ({market}) ===")
    generate_corr_matrices(market)
    print(f"=== Step 3/3: preprocessed .pkl samples ({market}) ===")
    build_pkls(market)
    print(f"=== DONE: {market} ===")
