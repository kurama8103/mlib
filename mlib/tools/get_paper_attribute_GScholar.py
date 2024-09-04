# %%
# Google Scholarから論文の基本情報をpython で取得する
# https://qiita.com/kuto/items/9730037c282da45c1d2b

import sys
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re


def get_search_results(keyword, number=10):
    # Not to omit columns width
    pd.set_option("display.max_colwidth", None)

    columns = ["rank", "title", "writer", "year", "citations", "url"]
    df = pd.DataFrame(columns=columns)  # 表の作成
    html_doc = requests.get(
        "https://scholar.google.co.jp/scholar?hl=ja&as_sdt=0%2C5&num="
        + str(number)
        + "&q="
        + keyword
    ).text
    soup = BeautifulSoup(html_doc, "html.parser")  # BeautifulSoupの初期化
    tags1 = soup.find_all("h3", {"class": "gs_rt"})  # title&url
    tags2 = soup.find_all("div", {"class": "gs_a"})  # writer&year
    tags3 = soup.find_all(text=re.compile("引用元"))  # citation

    rank = 1
    l = []
    for tag1, tag2, tag3 in zip(tags1, tags2, tags3):
        title = tag1.text.replace("[HTML]", "")
        if tag1.select("a") == []:
            url = ""
        else:
            url = tag1.select("a")[0].get("href")
        writer = tag2.text
        writer = re.sub(r"\d", "", writer)
        year = tag2.text
        year = re.sub(r"\D", "", year)
        citations = tag3.replace("引用元", "")
        l.append([rank, title, writer, year, citations, url])
        rank + 1

    df = pd.DataFrame(l, columns=columns)
    return df


if __name__ == "__main__":
    if "ipykernel_launcher.py" in sys.argv[0]:
        keyword = "GPT"
    else:
        keyword = sys.argv[1]
    print(get_search_results(keyword).to_json())

# %%
