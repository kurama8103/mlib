# %%
import subprocess
import sys
import warnings

import pandas as pd
import quantstats as qs
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


def quantstats_html(
    filepath_csv: str, open_html: bool = True, no_benchmark: bool = True
) -> None:
    """
    first column is used benchmark.
    """
    qs.extend_pandas()
    df = pd.read_csv(filepath_csv, index_col=0, parse_dates=True)
    if df.min().min() <= 0:
        pass
    else:
        df = df.pct_change()

    for c in tqdm(df.columns):
        x = df[c].dropna()
        if (df.iloc[:, 0].name == c) or no_benchmark:
            bm = None
        else:
            bm = df.iloc[:, 0]
        fn = "qs_" + x.name + x.index[-1].strftime("_%Y%m%d") + ".html"
        print("", c)
        qs.reports.html(
            returns=x, rf=0.0, title=c, benchmark=bm, download_filename=fn, output="./"
        )
        if open_html == True:
            subprocess.call(["open", fn])
    return None


if __name__ == "__main__":
    if "ipykernel_launcher.py" in sys.argv[0]:
        filepath_csv = "../data/test.csv"
    else:
        filepath_csv = sys.argv[1]
    quantstats_html(filepath_csv, open_html=True)

# %%
