#!/usr/bin/env python3
"""Aggregate the hourly pilot into explicit left-labelled six-hour intervals.

AORC precipitation: preceding-hour accumulation; sum labels start+1h..end.
NWM and other fields: arithmetic mean of hourly samples start..end-1h.
Six finite hourly records are required per variable/node. No gap filling.
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
             "UGRD_10maboveground", "VGRD_10maboveground", "streamflow"]


def aggregate(values, precipitation=False):
    if values.shape[0] % 6:
        raise ValueError("Hourly block length is not divisible by six")
    blocks = values.reshape(-1, 6, values.shape[1])
    valid = np.isfinite(blocks)
    count = valid.sum(axis=1).astype(np.uint8)
    result = np.where(valid, blocks, 0).sum(axis=1, dtype=np.float64)
    if not precipitation:
        result /= 6
    result[count != 6] = np.nan
    return result.astype(np.float32), count


def boundary_precipitation(root, date, ids, available):
    mapping = root / "processed/aorc_mapping"
    weights = load_npz(mapping / "aorc_to_nodes_weights.npz").tocsr()
    with np.load(mapping / "grid_and_node_order.npz") as order:
        assert np.array_equal(order["node_comid"], ids)
        latitude, longitude = order["latitude"].copy(), order["longitude"].copy()
    first = date.to_period("M").start_time
    last = first + pd.offsets.MonthBegin(1)
    stamp = f"{first:%Y%m%dT%H}_{last:%Y%m%dT%H}"
    path = root / "raw/aorc" / str(first.year) / f"tennessee_aorc_APCP_surface_{stamp}.nc"
    with xr.open_dataset(path, engine="netcdf4") as ds:
        assert np.array_equal(ds.latitude.values, latitude)
        assert np.array_equal(ds.longitude.values, longitude)
        matches = np.flatnonzero(pd.DatetimeIndex(ds.time.values) == date)
        assert len(matches) == 1, "Boundary precipitation timestamp absent or duplicated"
        field = ds.APCP_surface.isel(time=int(matches[0])).transpose("latitude", "longitude")
        raw = np.asarray(field.values, dtype=np.float32).reshape(-1)
        units = field.attrs.get("units", "")
    invalid = ~np.isfinite(raw) | (raw < 0)
    safe = np.where(invalid, 0, raw)
    values = np.asarray(weights @ safe).astype(np.float32)
    support = weights.copy()
    support.data[:] = 1
    values[np.asarray(support @ invalid.astype(np.float32)) > 0] = np.nan
    values[~available] = np.nan
    return values, str(path), units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/tva_data"))
    parser.add_argument("--month", default="2018-01")
    args = parser.parse_args()
    first = pd.Timestamp(args.month + "-01")
    last = first + pd.offsets.MonthBegin(1)
    hourly_times = pd.date_range(first, last, freq="h", inclusive="left").as_unit("ns")
    six_times = pd.date_range(first, last, freq="6h", inclusive="left").as_unit("ns")
    stamp = f"{first:%Y%m%dT%H}_{last:%Y%m%dT%H}"
    pilot = args.root / "processed/pilot_hourly" / args.month
    out = args.root / "processed/pilot_six_hour" / args.month
    out.mkdir(parents=True, exist_ok=True)
    with np.load(pilot / "node_order_and_forcing_mask.npz") as order:
        ids = order["node_comid"].copy()
        available = order["forcing_available"].astype(bool)
    nodes = pd.read_parquet(args.root / "processed/graph_candidate/nodes.parquet")
    assert np.array_equal(nodes.sort_values("node_index").comid.values, ids)
    sources = {v: pilot / f"tennessee_{v}_{stamp}_hourly_nodes.nc" for v in VARIABLES}
    units = {}
    for name, path in sources.items():
        with xr.open_dataset(path, engine="netcdf4") as ds:
            assert pd.DatetimeIndex(ds.time.values).as_unit("ns").equals(hourly_times), f"{path}: hourly timestamps differ"
            assert np.array_equal(ds.feature_id.values, ids)
            assert set(ds[name].dims) == {"time", "feature_id"}
            units[name] = ds[name].attrs.get("units", "")
            assert units[name], f"Missing units for {name}"
    accepted_precip_units = {"kg/m^2", "kg m-2", "kg m^-2", "kg m**-2", "mm"}
    assert units["APCP_surface"] in accepted_precip_units, "Unknown precipitation units"
    boundary, boundary_path, boundary_units = boundary_precipitation(args.root, last, ids, available)
    assert boundary_units == units["APCP_surface"], "Precipitation units changed at month boundary"

    destination = out / f"tennessee_{stamp}_six_hour_nodes.nc"
    partial = destination.with_suffix(".nc.part")
    rows = []
    with netCDF4.Dataset(partial, "w") as nc:
        nc.createDimension("time", len(six_times))
        nc.createDimension("feature_id", len(ids))
        nc.createDimension("bounds", 2)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "hours since 1970-01-01 00:00:00"
        t.calendar = "standard"
        t.bounds = "time_bounds"
        starts = six_times.asi8 / 3.6e12
        t[:] = starts
        bounds = nc.createVariable("time_bounds", "f8", ("time", "bounds"))
        bounds.units, bounds.calendar = t.units, t.calendar
        bounds[:] = np.column_stack([starts, starts + 6])
        nc.createVariable("feature_id", "i8", ("feature_id",))[:] = ids
        nc.createVariable("forcing_available", "u1", ("feature_id",))[:] = available.astype("uint8")
        nc.interval_label = "start; UTC assumed from source archive time convention"
        nc.interval_definition = "[time, time + 6 hours)"
        nc.precipitation_definition = "Sum of preceding-hour accumulations labelled time+1h through time+6h"
        nc.discharge_definition = "Mean of six hourly NWM samples; approximation, not an exact time integral"
        nc.missing_policy = "All six hourly records must be finite; otherwise NaN"
        nc.normalization = "None"
        nc.precipitation_boundary_source = boundary_path
        chunks = (min(4, len(six_times)), min(2048, len(ids)))

        for name, path in sources.items():
            precip = name == "APCP_surface"
            variable = nc.createVariable(name, "f4", ("time", "feature_id"),
                fill_value=np.float32(np.nan), zlib=True, complevel=2,
                shuffle=True, chunksizes=chunks)
            variable.units = "mm" if precip else units[name]
            variable.cell_methods = "time: sum" if precip else "time: mean"
            count_var = nc.createVariable(name + "_valid_hours", "u1",
                ("time", "feature_id"), zlib=True, complevel=2, chunksizes=chunks)
            count_var.long_name = "Number of finite hourly records contributing to interval (0 to 6)"
            count_var.units = "1"
            variable.ancillary_variables = name + "_valid_hours"
            stats = {"variable": name, "units": variable.units,
                     "valid_intervals": 0, "masked_intervals": 0,
                     "partial_intervals": 0, "minimum": np.inf, "maximum": -np.inf}
            hourly_total = np.zeros(len(ids), dtype=np.float64)
            aggregate_total = np.zeros(len(ids), dtype=np.float64)
            full_month_valid = np.ones(len(ids), dtype=bool)
            with xr.open_dataset(path, engine="netcdf4") as ds:
                field = ds[name].transpose("time", "feature_id")
                for start in tqdm(range(0, len(hourly_times), 24), desc=name):
                    stop = min(start + 24, len(hourly_times))
                    shift = 1 if precip else 0
                    values = np.asarray(field.isel(time=slice(start+shift, stop+shift)).values,
                                        dtype=np.float32)
                    if precip and stop == len(hourly_times):
                        values = np.concatenate([values, boundary[None, :]], axis=0)
                    result, counts = aggregate(values, precipitation=precip)
                    begin_interval = start // 6
                    end_interval = stop // 6
                    variable[begin_interval:end_interval] = result
                    count_var[begin_interval:end_interval] = counts
                    finite = np.isfinite(result)
                    stats["valid_intervals"] += int(finite.sum())
                    stats["masked_intervals"] += int((~finite).sum())
                    stats["partial_intervals"] += int(((counts > 0) & (counts < 6)).sum())
                    if finite.any():
                        stats["minimum"] = min(stats["minimum"], float(result[finite].min()))
                        stats["maximum"] = max(stats["maximum"], float(result[finite].max()))
                    if precip:
                        full_month_valid &= np.isfinite(values).all(axis=0)
                        hourly_total += np.where(np.isfinite(values), values, 0).sum(axis=0, dtype=np.float64)
                        aggregate_total += np.where(finite, result, 0).sum(axis=0, dtype=np.float64)
            if precip:
                if not full_month_valid.any():
                    raise RuntimeError("No fully valid precipitation nodes for conservation check")
                errors = np.abs(hourly_total[full_month_valid] - aggregate_total[full_month_valid])
                assert np.allclose(hourly_total[full_month_valid], aggregate_total[full_month_valid],
                                   rtol=1e-6, atol=1e-5), "Precipitation conservation failed"
                stats["conservation_check_nodes"] = int(full_month_valid.sum())
                stats["maximum_monthly_precipitation_difference_mm"] = float(errors.max())
            for key in ["minimum", "maximum"]:
                if not np.isfinite(stats[key]):
                    stats[key] = None
            rows.append(stats)
    partial.replace(destination)
    pd.DataFrame(rows).to_csv(out / "six_hour_variable_audit.csv", index=False)
    report = {"month": args.month, "six_hour_intervals": len(six_times),
        "graph_nodes": len(ids), "forcing_available_nodes": int(available.sum()),
        "node_order_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "precipitation_boundary_source": boundary_path,
        "output": str(destination), "variable_results": rows}
    (out / "six_hour_report.json").write_text(json.dumps(report, indent=2))
    print(pd.DataFrame(rows)[["variable", "units", "valid_intervals", "masked_intervals",
        "partial_intervals", "minimum", "maximum"]].to_string(index=False))
    print(f"\nSix-hour intervals: {len(six_times)}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
