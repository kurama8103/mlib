from os import path
from sqlite3 import connect
import investpy
import pandas as pd
import quandl
from mlib.util.sqls import execute_sql

db_path = path.abspath("timeseries.db")


def get_data_quandl(code, start=None, end=None):
    """
    btc : BITFINEX/BTCJPY
    """
    if start == None and end == None:
        df_tmp = execute_sql(
            db_path, '''select * from timeseries where code="{}"'''.format(code)
        )
        df = pd.DataFrame(df_tmp[0], columns=df_tmp[1]).set_index("date")["value"]
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
    else:
        df = quandl.get(dataset=code, start_date=start, end_date=end)
    df.name = code
    return df


def get_investpy_ETFs(from_date, to_date):
    etfs = ["VT", "BNDX", "VNQ", "GSG", "HYG", "TIP"]
    l = investpy.etfs.get_etfs(country="united states")
    names = l.query("symbol in @etfs")["name"]
    etf_dict = dict(zip(etfs, names))
    df_etf = dict()
    from_date, to_date = pd.to_datetime([from_date, to_date]).strftime("%d/%m/%Y")
    for k, v in etf_dict.items():
        df_etf[k] = investpy.get_etf_historical_data(
            etf=v,
            country="united states",
            from_date=from_date,
            to_date=to_date,
            interval="Daily",
        )["Close"]
    return df_etf


def init_db(db_path):
    sql = """CREATE TABLE timeseries (date int, code text, value numeric, PRIMARY KEY(date, code))"""
    execute_sql(db_path, sql)


def insert_timeseries_quandl(db_path, df_date_code_value):
    df = pd.DataFrame(df_date_code_value).stack().reset_index()
    df.iloc[:, 0] = df.iloc[:, 0].dt.strftime("%Y%m%d")

    con = connect(db_path)
    cur = con.cursor()
    # df_date_code_value.columns = ['date', 'code', 'value']
    # df_date_code_value.to_sql('timeseries', con, index=False, if_exists='append')

    sql = """INSERT or IGNORE INTO timeseries (date, code, value) values (?,?,?)"""
    cur.executemany(sql, df.values)
    con.commit()
    con.close()


def get_data_db(db_path="timeseries.db", sql="select * from timeseries"):
    df_ = execute_sql(db_path, sql)[0]
    df = pd.DataFrame(df_).pivot(0, 1)[2].reset_index().set_index(0)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.columns.name = None
    df.index.name = "date"
    return df
