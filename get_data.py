import os
import io
import zipfile
import urllib.request
import pandas as pd

OUT = "data/data_banknote_authentication.txt"
os.makedirs("data", exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=60).read()


def from_uci_txt():
    raw = fetch("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
    return pd.read_csv(io.BytesIO(raw), header=None)


def from_uci_zip():
    raw = fetch("https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip")
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.endswith(".txt")][0]
    return pd.read_csv(z.open(name), header=None)


def from_openml():
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=1462, as_frame=True)
    X = d.data.astype(float).reset_index(drop=True)
    y = d.target.reset_index(drop=True)
    # the bigger class (762 rows) is the genuine one -> 0, fake -> 1
    genuine_label = y.value_counts().idxmax()
    labels = (y != genuine_label).astype(int)
    df = X.copy()
    df.columns = [0, 1, 2, 3]
    df[4] = labels
    return df


sources = [("UCI text file", from_uci_txt),
           ("UCI zip file", from_uci_zip),
           ("OpenML", from_openml)]

ok = False
for name, fn in sources:
    try:
        print("Trying:", name)
        df = fn()
        if df.shape[1] == 5 and len(df) > 1000:
            df.to_csv(OUT, header=False, index=False)
            print("SUCCESS from", name, "->", df.shape, "saved to", OUT)
            ok = True
            break
    except Exception as e:
        print("  failed:", type(e).__name__, str(e)[:100])

if not ok:
    print("ALL SOURCES FAILED - use the manual download method.")
