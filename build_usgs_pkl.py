"""Build data/frenchbroad/usgs_data_dv.pkl from the cached nwis_iv parquet files,
in the exact shape nwmutils.getUSGSData() expects, so loadrivernet.py's
reload_usgs=False path finds it instead of trying to hit the network.

nwis_iv holds instantaneous (iv) values in cfs; getUSGSData's 'dv' path expects
a daily-mean 'Q' column in m3/s, one DataFrame per USGS site, keyed by site id
(matching the values in gageid.pkl / self.gageIDset).
"""
import glob
import os
import pickle as pkl

import pandas as pd

PROJECT = os.environ["PROJECT"]
SRC = f"{PROJECT}/fm_data/french_broad_06010105/nwis_iv"
DST = f"{PROJECT}/GraphRivers/data/frenchbroad"
os.makedirs(DST, exist_ok=True)

CFS2M3S = 0.028316847

files = sorted(glob.glob(f"{SRC}/06010105_nwis_iv_*.parquet"))
print(f"found {len(files)} nwis_iv files")

frames = []
for f in files:
    df = pd.read_parquet(f)
    frames.append(df)
allrec = pd.concat(frames, ignore_index=True)
allrec["time"] = pd.to_datetime(allrec["time"], utc=True)
print("total records:", len(allrec), "sites:", allrec["site_no"].nunique())

usgsDataDict = {}
for site, g in allrec.groupby("site_no"):
    g = g.set_index("time").sort_index()
    daily = g["value"].resample("24H").mean().to_frame("Q")
    daily["Q"] = daily["Q"] * CFS2M3S
    usgsDataDict[site] = daily
    print(f"site {site}: {len(daily)} daily records, "
          f"{daily.index.min().date()} to {daily.index.max().date()}")

out = f"{DST}/usgs_data_dv.pkl"
pkl.dump(usgsDataDict, open(out, "wb"))
print("wrote", out, "-", len(usgsDataDict), "sites")
