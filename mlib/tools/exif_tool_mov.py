#!/Users/user/.pyenv/versions/3.8.0/bin/python
# %%
import exiftool
import pandas as pd
from pprint import pprint as print
import glob
import os
import subprocess
import sys
from tqdm.auto import tqdm


def extract_keys(dict_: dict, keyword: str):
    return {k: v for k, v in dict_.items() if keyword in k}


def convert_and_set_metadata_MOV(files: list, convert: bool = False):
    for f in files:
        a, b = os.path.splitext(f)
        if not b.lower() in [".mov"]:
            break

        g = a + "C" + b

        if convert:
            subprocess.run(
                [
                    "avconvert",
                    "--preset",
                    "PresetHEVC1920x1080",
                    "--source",
                    f,
                    "--output",
                    g,
                    "--progress",
                    # "--replace",
                ],
            )

        with exiftool.ExifToolHelper() as et:
            data = et.get_metadata(f)
            if convert:
                data_a = et.get_metadata(g)

                tags = dict()
                tags["before"] = extract_keys(data[0], "Date")
                tags["change"] = extract_keys(data_a[0], "Date")
                df = pd.DataFrame().from_dict(tags, orient="columns")
                df["match"] = df["before"] == df["change"]
                # print(df)

                et.set_tags(
                    g, tags=tags["before"], params=["-P", "-overwrite_original"]
                )
                df["after"] = extract_keys(et.get_metadata(g)[0], "Date")
                print(df)
            else:
                print(extract_keys(data[0], "Date"))


if __name__ == "__main__":
    arg = sys.argv
    convert_and_set_metadata_MOV(arg[1:])
else:
    files = glob.glob(os.getcwd() + "/*[0-9].MOV")
    convert_and_set_metadata_MOV(files)

# %%
