#!/usr/bin/env python3
"""Prepare one hourly month of catchment AORC and NWM pretraining labels.

No normalization, gap filling, six-hour aggregation, or training is performed.
Missing-catchment nodes stay in the graph with NaN forcing. A catchment value
is missing whenever any positively weighted grid cell is missing/invalid.
"""
import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import load_npz
from tqdm import tqdm

VARIABLES = ["APCP_surface", "TMP_2maboveground", "SPFH_2maboveground",
             "PRES_surface", "DSWRF_surface", "DLWRF_surface",
             "UGRD_10maboveground", "VGRD_10maboveground"]


def spatial_average(values, weights, support, available, nonnegative=False):
    """values: (hours, flattened grid cells); no missing-cell renormalization."""
    invalid = ~np.isfinite(values)
    if nonnegative:
        invalid |= values < 0
    safe = np.where(invalid, 0, values).astype(np.float32)
    averaged = np.asarray(weights @ safe.T).T.astype(np.float32)
    any_missing = np.asarray(support @ invalid.astype(np.float32).T).T > 0
    averaged[any_missing] = np.nan
    averaged[:, ~available] = np.nan
    return averaged, int(invalid.sum())


def open_output(path, times, ids, name, attrs):
    handle = netCDF4.Dataset(path, "w")
    handle.createDimension("time", len(times))
    handle.createDimension("feature_id", len(ids))
    t = handle.createVariable("time", "f8", ("time",))
    t.units = "hours since 1970-01-01 00:00:00"
    t.calendar = "standard"
    t[:] = (times.asi8 / 3.6e12).astype(np.float64)
    fid = handle.createVariable("feature_id", "i8", ("feature_id",))
    fid[:] = ids
    chunks = (min(24, len(times)), min(2048, len(ids)))
    v = handle.createVariable(name, "f4", ("time", "feature_id"),
        fill_value=np.float32(np.nan), zlib=True, complevel=2,
        shuffle=True, chunksizes=chunks)
    for key in ["units", "long_name", "standard_name"]:
        if key in attrs:
            v.setncattr(key, attrs[key])
    handle.setncattr("timestamp_policy", "Source timestamps preserved; interval interpretation pending")
    return handle, v


def statistics(values, stats):
    finite = np.isfinite(values)
    stats["finite_values"] += int(finite.sum())
    stats["missing_values"] += int(values.size - finite.sum())
    if finite.any():
        stats["minimum"] = min(stats["minimum"], float(values[finite].min()))
        stats["maximum"] = max(stats["maximum"], float(values[finite].max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    parser.add_argument("--month", default="2018-01")
    parser.add_argument("--block-hours", type=int, default=24)
    args = parser.parse_args()
    if args.block_hours < 1:
        parser.error("--block-hours must be positive")
    first = pd.Timestamp(args.month + "-01")
    last = first + pd.offsets.MonthBegin(1)
    times = pd.date_range(first, last, freq="h", inclusive="left")
    stamp = f"{first:%Y%m%dT%H}_{last:%Y%m%dT%H}"
    mapping = args.root / "processed/aorc_mapping"
    weights = load_npz(mapping / "aorc_to_nodes_weights.npz").tocsr()
    with np.load(mapping / "grid_and_node_order.npz") as order:
        latitude = order["latitude"].copy()
        longitude = order["longitude"].copy()
        ids = order["node_comid"].copy()
        available = order["forcing_available"].astype(bool)
    nodes = pd.read_parquet(args.root / "processed/graph_candidate/nodes.parquet")
    nodes = nodes.sort_values("node_index")
    assert np.array_equal(ids, nodes.comid.to_numpy()), "Graph/mapping node order differs"
    assert weights.shape == (len(ids), len(latitude)*len(longitude))
    assert np.isfinite(weights.data).all() and (weights.data >= 0).all()
    row_sum = np.asarray(weights.sum(axis=1)).ravel()
    assert np.allclose(row_sum[available], 1, atol=1e-6)
    assert np.all(row_sum[~available] == 0)
    support = weights.copy()
    support.data[:] = 1

    inputs = [(v, args.root / "raw/aorc" / str(first.year)
        / f"tennessee_aorc_{v}_{stamp}.nc") for v in VARIABLES]
    inputs.append(("streamflow", args.root / "raw/nwm"
        / f"tennessee_nwm_{stamp}.nc"))
    # Preflight all inputs before writing any processed file.
    for name, source in inputs:
        if not source.is_file():
            raise FileNotFoundError(source)
        with xr.open_dataset(source, engine="netcdf4") as ds:
            assert name in ds, f"{source}: variable missing"
            assert pd.DatetimeIndex(ds.time.values).equals(times), f"{source}: hourly time mismatch"
            assert ds[name].attrs.get("units"), f"{source}: units missing"
            if name != "streamflow":
                assert set(ds[name].dims) == {"time", "latitude", "longitude"}
                assert np.array_equal(ds.latitude.values, latitude), "Latitude order differs"
                assert np.array_equal(ds.longitude.values, longitude), "Longitude order differs"
            else:
                assert set(ds[name].dims) == {"time", "feature_id"}
                source_ids = ds.feature_id.values
                assert len(np.unique(source_ids)) == len(source_ids)
                assert set(source_ids) == set(ids), "NWM node IDs differ"

    out = args.root / "processed/pilot_hourly" / args.month
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, source in inputs:
        destination = out / f"tennessee_{name}_{stamp}_hourly_nodes.nc"
        partial = destination.with_suffix(".nc.part")
        stats = {"variable": name, "finite_values": 0, "missing_values": 0,
                 "minimum": np.inf, "maximum": -np.inf,
                 "invalid_input_values": 0}
        with xr.open_dataset(source, engine="netcdf4") as ds:
            field = ds[name]
            if name == "streamflow":
                field = field.sel(feature_id=ids).transpose("time", "feature_id")
            else:
                field = field.transpose("time", "latitude", "longitude")
            attrs = field.attrs.copy()
            handle, variable = open_output(partial, times, ids, name, attrs)
            try:
                handle.source_file = str(source)
                handle.source_size_bytes = str(source.stat().st_size)
                handle.source_mtime_ns = str(source.stat().st_mtime_ns)
                if name != "streamflow":
                    handle.spatial_method = "Area-weighted incremental catchments; strict missing-cell mask"
                else:
                    handle.target_source = "NWM simulations; pretraining labels, not observed truth"
                for start in tqdm(range(0, len(times), args.block_hours), desc=name):
                    stop = min(start + args.block_hours, len(times))
                    values = np.asarray(field.isel(time=slice(start, stop)).values,
                                        dtype=np.float32)
                    if name == "streamflow":
                        invalid = ~np.isfinite(values) | (values < 0)
                        stats["invalid_input_values"] += int(invalid.sum())
                        values[invalid] = np.nan
                        processed = values
                    else:
                        processed, invalid_count = spatial_average(
                            values.reshape(stop-start, -1), weights, support,
                            available, nonnegative=(name == "APCP_surface"))
                        stats["invalid_input_values"] += invalid_count
                    variable[start:stop, :] = processed
                    statistics(processed, stats)
            finally:
                handle.close()
        partial.replace(destination)
        stats["units"] = attrs["units"]
        stats["minimum"] = stats["minimum"] if np.isfinite(stats["minimum"]) else None
        stats["maximum"] = stats["maximum"] if np.isfinite(stats["maximum"]) else None
        stats["missing_fraction"] = stats["missing_values"] / (len(times)*len(ids))
        expected_unavailable = 0 if name == "streamflow" else len(times)*int((~available).sum())
        stats["expected_missing_without_catchment"] = expected_unavailable
        stats["additional_missing_values"] = stats["missing_values"] - expected_unavailable
        stats["output"] = str(destination)
        rows.append(stats)
        pd.DataFrame(rows).to_csv(out / "variable_audit.csv", index=False)
        print(f"Saved: {destination}", flush=True)

    np.savez_compressed(out / "node_order_and_forcing_mask.npz",
        node_comid=ids, forcing_available=available)
    report = {
        "month": args.month, "hourly_steps": len(times), "graph_nodes": len(ids),
        "nodes_with_catchment_forcing": int(available.sum()),
        "nodes_without_catchment_forcing": int((~available).sum()),
        "node_order_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "time_alignment": "Identical source timestamps; interval support still to verify",
        "missing_policy": "NaN; no interpolation or missing-cell renormalization",
        "negative_policy": "Negative precipitation/discharge masked; other variables unchanged",
        "normalization": "None", "training": "None", "six_hour_aggregation": "None",
        "variable_results": rows,
    }
    (out / "pilot_report.json").write_text(json.dumps(report, indent=2))
    print(pd.DataFrame(rows)[["variable", "units", "minimum", "maximum",
        "expected_missing_without_catchment", "additional_missing_values"]].to_string(index=False))
    print(f"\nPilot prepared: {out}", flush=True)


if __name__ == "__main__":
    main()
