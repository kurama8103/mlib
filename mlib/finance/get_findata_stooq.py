# %%
import pandas_datareader.data as web
import pandas as pd
import datetime as dt
from tqdm.auto import tqdm
import subprocess


def get_findata_stooq(timedelta=100):
    indices_d = {
        #     '^GSPC': 'S&P 500',
        '^SPX': 'S&P 500',
        # '^IXIC': 'NASDAQ',
        '^TPX': 'TOPIX',
        #     '^N225': 'Nikkei 225',
        #     '^TNX': 'Treasury Yield 10 Years',
        'VTI': 'Vanguard Total Stock Market Index Fund ETF',
        'VGK': 'Vanguard European Stock Index Fund ETF',
        #         'EWJ': 'iShares MSCI Japan ETF',
        'VWO': 'Vanguard Emerging Markets Stock Index Fund ETF',
        'BND': 'Vanguard Total Bond Market Index Fund ETF',
        #         'BNDX': 'Vanguard Total International Bond Index Fund ETF',
        #     'AGG':'iShares Core US Aggregate Bond ETF',
        'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF',
        'TIP': 'iShares TIPS Bond ETF',
        #         'VMBS': 'Vanguard Mortgage-Backed Secs Idx Fund ETF',
        # 'RX=F':'Dow Jones Real Estate Futures',
        # '^REI': 'Dow Jones Equity All REIT Index',
        'IYR': 'iShares US Real Estate ETF',
        #     'RWR':'SPDR Dow Jones REIT ETF',
        #     'RWX':'SPDR Dow Jones International Real Estate ETF'
        #     'VNQ':'Vanguard Real Estate Index Fund ETF ',
        # 'CL=F': 'Crude Oil',
        # 'GC=F': 'Gold',
        # '^SPGSCI': 'S&P GSCI Index ',
        'GSG': 'iShares S&P GSCI Commodity-Indexed Trust',
        # 'TPY=F': 'Yen Denominated TOPIX Futures',
        #     'BTC-USD' : 'Bitcoin USD',
        #     '^VIX':'CBOE Volatility Index'
    }

    indices_ds = {
        'IVV': 'US_equity',
        'IDEV': 'DEV_equity',
        'IEMG': 'EM_equity',
        'AGG': 'US_bond',
        'IAGG': 'DEV_bond',
        'EMB': 'EM_bond',
        #         'MBB':'US_MBS',
        'TIP': 'US_TIPS',
        'REET': 'G_reit',
        'IAU': 'gold',
        'GSG': 'Commodity',
        'BITO': 'bitcoin'
    }
    indices_core = {
        'VT': 'equity',
        'BNDX': 'bond',
        'REET': 'reit',
        'GSG': 'Commodity',
        'BITO': 'bitcoin',
        'IAU': 'gold',
        'FXY':'JPY'
    }

    df = pd.DataFrame()
    dt_st = (pd.to_datetime(dt.datetime.today()) -
             dt.timedelta(timedelta)).strftime('%Y-%m-%d')

    for x in tqdm(indices_core.keys()):
        try:
            df_ = pd.DataFrame(web.DataReader(
                x, 'stooq', start=dt_st)['Close'])
            df_.columns = [x]
            df = pd.concat([df, df_], axis=1)
        except:
            print('eroor: ', x)

    df.columns = [s.replace('^', '').replace('=F', '')
                  for s in df.columns.values]
    df = df.sort_index()
    return df


if __name__ == '__main__':
    df = get_findata_stooq()
    df.to_csv('get_stooq.csv')
    df.to_pickel('get_stooq.pickel')
    # subprocess.call(['open', 'get_stooq.csv'])
# %%
