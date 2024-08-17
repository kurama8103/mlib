#!/usr/bin/python
# -*- coding: utf-8 -*-
# %%
import pandas as pd

# url = 'https://www.jpx.co.jp/markets/indices/topix/tvdivq00000030ne-att/TOPIX_weight_jp.xlsx'
url = (
    "https://www.jpx.co.jp/markets/indices/topix/tvdivq00000030ne-att/topixweight_j.csv"
)


def main(url, file_name="out.csv"):
    df = pd.read_csv(url, encoding="cp932", dtype="object")
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    main(url=url)

# %%
