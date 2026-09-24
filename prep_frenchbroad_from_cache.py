#!/usr/bin/env python3
"""Build adjinfo2.1.pkl / nwmdfv21_24H.pkl / era5forcing*.pkl / gageid.pkl
for French Broad (06010105) from the pre-downloaded cache, bypassing
genrivernet.py's NHDPlus SQLite + NWM/AORC download path entirely.

Run on a compute node (not the login node) -- builds ~1-2GB intermediate
tables per forcing year. e.g.:
  interact -p RM-shared --ntasks-per-node=8 --mem=32G -t 04:00:00
"""
import glob, os, re, pickle as pkl
import numpy as np
import pandas as pd
import xarray as xr

PROJECT = os.environ["PROJECT"]
SRC = f"{PROJECT}/fm_data/french_broad_06010105"
DST = f"{PROJECT}/GraphRivers/data/frenchbroad"
os.makedirs(DST, exist_ok=True)

# ---------- 1. topology (fixes the node order used by EVERYTHING below) ----------
nodes = pd.read_parquet(f"{SRC}/graph/06010105_nodes.parquet")
edges = pd.read_parquet(f"{SRC}/graph/06010105_edges.parquet")

comid_order = nodes["comid"].tolist()
node_dict = {c: i for i, c in enumerate(comid_order)}
num_nodes = len(comid_order)

edges = edges[edges["comid"].isin(node_dict) & edges["tocomid"].isin(node_dict)]
start_node = [node_dict[c] for c in edges["comid"]]
end_node   = [node_dict[c] for c in edges["tocomid"]]

len_map = dict(zip(nodes["comid"], nodes["lengthkm"]))
edge_len = [len_map[c] for c in edges["comid"]]
da_map = dict(zip(nodes["comid"], nodes["areasqkm"]))
drainage_area = [da_map[c] for c in comid_order]
ftype = [0] * len(start_node)  # no StreamRiver/ArtificialPath tag available;
                                 # harmless since connect_exta_path=False in config

resDict = {
    "num_nodes": num_nodes, "start_node": start_node, "end_node": end_node,
    "node_dict": node_dict, "edge_len": edge_len, "drainage_area": drainage_area,
    "ftype": ftype, "extra_path": [], "extra_path_nodes": [],
}
pkl.dump(resDict, open(f"{DST}/adjinfo2.1.pkl", "wb"))
print("wrote adjinfo2.1.pkl, num_nodes =", num_nodes)

# ---------- 2. streamflow -> nwmdf, columns in the SAME comid_order ----------
sf_files = sorted(glob.glob(f"{SRC}/streamflow/06010105_all_reaches_streamflow_*.nc"))
sf_years = [int(re.search(r"(\d{4})\.nc$", f).group(1)) for f in sf_files]
print("streamflow years found:", sf_years)

sf_daily = []
gage_id_map = {}
for f in sf_files:
    ds = xr.open_dataset(f)
    df = ds["streamflow"].to_pandas().reindex(columns=comid_order)
    sf_daily.append(df.resample("24H").mean())   # resample immediately, keep memory bounded
    if not gage_id_map:
        gid = ds["gage_id"].values.astype(str)
        for fid, g in zip(ds["feature_id"].values, gid):
            g = g.strip()
            if g:
                gage_id_map[int(fid)] = g
    ds.close()

nwmdf = pd.concat(sf_daily).sort_index()
print("nwmdf shape (daily):", nwmdf.shape)
pkl.dump(nwmdf, open(f"{DST}/nwmdfv21_24H.pkl", "wb"))
print("reaches with a gage_id:", len(gage_id_map), gage_id_map)

# ---------- 3. ERA5-Land forcing -> dict[comid] = DataFrame(8 vars, daily) ----------
fc_files = sorted(glob.glob(f"{SRC}/forcing/06010105_forcing_era5land_*.parquet"))
fc_years = [int(re.search(r"(\d{4})\.parquet$", f).group(1)) for f in fc_files]
use_years = sorted(set(sf_years) & set(fc_years))
print("forcing years found:", fc_years, "| used (intersect w/ streamflow):", use_years)

FORCING_VARS = ["t2m", "d2m", "tp", "sp", "u10", "v10", "ssrd", "strd"]
var_daily = {v: [] for v in FORCING_VARS}
for f in fc_files:
    yr = int(re.search(r"(\d{4})\.parquet$", f).group(1))
    if yr not in use_years:
        continue
    df = pd.read_parquet(f)
    wide = df.pivot(index="time", columns="comid", values=FORCING_VARS)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([FORCING_VARS, comid_order]))
    daily = wide.resample("24H").mean()
    for v in FORCING_VARS:
        var_daily[v].append(daily[v])
    print("forcing year", yr, "done")

var_full = {v: pd.concat(var_daily[v]).sort_index() for v in FORCING_VARS}
forcingDict = {c: pd.DataFrame({v: var_full[v][c] for v in FORCING_VARS}) for c in comid_order}

pkl.dump(forcingDict, open(f"{DST}/era5forcing_full_nwm2.1_24H.pkl", "wb"))
print("wrote era5forcing pkl,", len(forcingDict), "comids")

# ---------- 4. gage mapping, cross-checked against nwis_iv site numbers on disk ----------
iv_files = sorted(glob.glob(f"{SRC}/nwis_iv/06010105_nwis_iv_*.parquet"))
have_sites = set()
for f in iv_files:
    have_sites.update(pd.read_parquet(f, columns=["site_no"])["site_no"].unique().tolist())

final_gage = {}
for comid, site in gage_id_map.items():
    if site in have_sites:
        final_gage[comid] = site
    else:
        print("!! gage_id on reach not found in nwis_iv cache:", comid, repr(site))

pkl.dump(final_gage, open(f"{DST}/gageid.pkl", "wb"))
print("\nfinal comid -> usgs gage mapping (paste into FrenchBroad class):")
print("self.obsComIDset =", list(final_gage.keys()))
print("self.gageIDset   =", list(final_gage.values()))