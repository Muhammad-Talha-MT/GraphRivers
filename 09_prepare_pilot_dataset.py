#!/usr/bin/env python3
"""Build a small six-hour pilot directly from raw AORC/NWM monthly files.
No training, interpolation, normalization, or realized future-rain inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import load_npz
from tqdm import tqdm

VARIABLES = ['APCP_surface', 'TMP_2maboveground', 'SPFH_2maboveground',
             'PRES_surface', 'DSWRF_surface', 'DLWRF_surface',
             'UGRD_10maboveground', 'VGRD_10maboveground']


def raw_path(root, name, month):
    end = month + pd.offsets.MonthBegin(1)
    stamp = f'{month:%Y%m%dT%H}_{end:%Y%m%dT%H}'
    if name == 'streamflow':
        return root / 'raw/nwm' / f'tennessee_nwm_{stamp}.nc'
    return root / 'raw/aorc' / str(month.year) / f'tennessee_aorc_{name}_{stamp}.nc'


def reduce_six(values, rain=False):
    blocks = values.reshape(-1, 6, values.shape[-1])
    valid = np.isfinite(blocks)
    count = valid.sum(axis=1).astype('uint8')
    safe = np.where(valid, blocks, 0)
    result = safe.sum(axis=1) if rain else safe.sum(axis=1) / 6
    result[count != 6] = np.nan
    return result.astype('float32'), count


def map_values(values, weights, support, available, rain=False):
    invalid = ~np.isfinite(values)
    if rain:
        invalid |= values < 0
    mapped = np.asarray(weights @ np.where(invalid, 0, values).T).T
    missing = np.asarray(support @ invalid.astype('float32').T).T > 0
    mapped[missing] = np.nan
    mapped[:, ~available] = np.nan
    return mapped.astype('float32')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/tva_data'))
    p.add_argument('--pilot', default='rosn7')
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--stop', default='2018-07-01', help='Exclusive; month boundary')
    p.add_argument('--block-hours', type=int, default=24)
    args = p.parse_args()
    start, stop = pd.Timestamp(args.start), pd.Timestamp(args.stop)
    if start.day != 1 or stop.day != 1 or start != start.normalize() or stop != stop.normalize():
        p.error('Start and stop must be midnight month boundaries')
    if not start < stop or stop > pd.Timestamp('2019-01-01'):
        p.error('This training pilot must end by 2019-01-01, exclusively')
    if args.block_hours < 1:
        p.error('Block hours must be positive')
    graph = args.root / 'processed/pilot_graph' / args.pilot
    nodes = pd.read_parquet(graph / 'nodes.parquet').sort_values('node_index')
    ids = nodes.comid.to_numpy(dtype='int64')
    if len(np.unique(ids)) != len(ids) or not np.array_equal(nodes.node_index, np.arange(len(ids))):
        raise ValueError('Pilot node IDs/index are not unique and contiguous')
    mapping = args.root / 'processed/aorc_mapping'
    with np.load(mapping / 'grid_and_node_order.npz') as info:
        lat, lon = info['latitude'].copy(), info['longitude'].copy()
        all_ids = info['node_comid'].copy()
        indices = pd.Index(all_ids).get_indexer(ids)
        if (indices < 0).any():
            raise ValueError('Pilot IDs absent from mapping')
        available = info['forcing_available'][indices].astype(bool)
    weights = load_npz(mapping / 'aorc_to_nodes_weights.npz').tocsr()[indices]
    if weights.shape[1] != len(lat) * len(lon) or not weights.nnz:
        raise ValueError('Unexpected/empty grid mapping')
    rows, cols = np.divmod(weights.indices, len(lon))
    y0, y1, x0, x1 = int(rows.min()), int(rows.max())+1, int(cols.min()), int(cols.max())+1
    grid_columns = (np.arange(y0, y1)[:, None] * len(lon) + np.arange(x0, x1)).ravel()
    weights = weights[:, grid_columns].tocsr()
    sums = np.asarray(weights.sum(axis=1)).ravel()
    if not np.isfinite(weights.data).all() or (weights.data < 0).any() or not np.allclose(sums[available], 1, atol=1e-6) or (sums[~available] != 0).any():
        raise ValueError('Invalid spatial weights')
    support = weights.copy()
    support.data[:] = 1
    months = pd.date_range(start, stop, freq='MS', inclusive='left')
    sources = [(name, month) for month in months for name in VARIABLES + ['streamflow']]
    sources.append(('APCP_surface', stop))  # Final interval needs the following midnight rainfall.
    source_records = []
    for name, month in sources:
        path = raw_path(args.root, name, month)
        with xr.open_dataset(path, engine='netcdf4') as ds:
            if name not in ds or not ds[name].attrs.get('units'):
                raise ValueError(f'Missing variable/units: {path}')
            t = pd.DatetimeIndex(ds.time.values)
            expected = pd.date_range(month, month + pd.offsets.MonthBegin(1), freq='h', inclusive='left')
            if not t.equals(expected):
                raise ValueError(f'Hourly timestamps differ: {path}')
            if name == 'streamflow':
                if set(ds[name].dims) != {'time', 'feature_id'} or not pd.Index(ds.feature_id.values).is_unique or (pd.Index(ds.feature_id.values).get_indexer(ids) < 0).any():
                    raise ValueError(f'NWM node IDs/dimensions differ: {path}')
            else:
                if set(ds[name].dims) != {'time', 'latitude', 'longitude'} or not np.array_equal(ds.latitude.values, lat) or not np.array_equal(ds.longitude.values, lon):
                    raise ValueError(f'AORC grid differs: {path}')
                if name == 'APCP_surface' and ds[name].attrs['units'] not in {'kg/m^2', 'kg m-2', 'mm'}:
                    raise ValueError('Rainfall units require review')
        source_records.append({'file': str(path), 'size_bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns})

    output = args.root / 'processed/pilot_dataset' / args.pilot
    output.mkdir(parents=True, exist_ok=True)
    pieces, audits = [], []
    for month in months:
        end = month + pd.offsets.MonthBegin(1)
        hourly_time = pd.date_range(month, end, freq='h', inclusive='left')
        times = pd.date_range(month, end, freq='6h', inclusive='left')
        data = {}
        for name in tqdm(VARIABLES + ['streamflow'], desc=f'{month:%Y-%m}'):
            with xr.open_dataset(raw_path(args.root, name, month), engine='netcdf4') as ds:
                attrs = dict(ds[name].attrs)
                if name == 'streamflow':
                    field = ds[name].sel(feature_id=ids).transpose('time', 'feature_id')
                else:
                    field = ds[name].isel(latitude=slice(y0, y1), longitude=slice(x0, x1)).transpose('time', 'latitude', 'longitude')
                hourly = np.empty((len(hourly_time), len(ids)), dtype='float32')
                for offset in range(0, len(hourly_time), args.block_hours):
                    limit = min(offset + args.block_hours, len(hourly_time))
                    values = np.asarray(field.isel(time=slice(offset, limit)).values, dtype='float32')
                    if name == 'streamflow':
                        values[~np.isfinite(values) | (values < 0)] = np.nan
                        hourly[offset:limit] = values
                    else:
                        hourly[offset:limit] = map_values(values.reshape(limit-offset, -1), weights, support, available, name == 'APCP_surface')
            if name == 'APCP_surface':
                with xr.open_dataset(raw_path(args.root, name, end), engine='netcdf4') as ds:
                    values = ds[name].sel(time=end).isel(latitude=slice(y0, y1), longitude=slice(x0, x1)).transpose('latitude', 'longitude').values
                    tail = map_values(np.asarray(values, dtype='float32').reshape(1, -1), weights, support, available, True)
                hourly = np.concatenate([hourly[1:], tail], axis=0)
                attrs['units'] = 'mm'
            values, count = reduce_six(hourly, name == 'APCP_surface')
            data[name] = (('time', 'feature_id'), values, attrs)
            data[name + '_valid_hours'] = (('time', 'feature_id'), count, {'units': '1'})
            audits.append({'month': f'{month:%Y-%m}', 'variable': name, 'valid_intervals': int((count == 6).sum()), 'masked_intervals': int((count < 6).sum()), 'partial_intervals': int(((count > 0) & (count < 6)).sum())})
        piece = xr.Dataset(data, coords={'time': times, 'feature_id': ids})
        pieces.append(piece)
    combined = xr.concat(pieces, dim='time')
    times = pd.DatetimeIndex(combined.time.values)
    expected = pd.date_range(start, stop, freq='6h', inclusive='left')
    if not times.equals(expected):
        raise ValueError('Combined six-hour timeline differs')
    combined['forcing_available'] = ('feature_id', available)
    bounds = np.stack([times.values, (times + pd.Timedelta(hours=6)).values], axis=1)
    combined['time_bounds'] = (('time', 'bounds'), bounds)
    combined.time.attrs['bounds'] = 'time_bounds'
    combined.attrs.update({
        'interval_policy': 'Left-labelled [time,time+6h); rainfall sum of ending-hour labels time+1h..time+6h; other fields mean of samples time..time+5h',
        'discharge_policy': 'Mean of six hourly NWM samples; simulated pretraining labels, not observed truth',
        'missing_policy': 'All six samples required; NaN, no interpolation',
        'mapping_status': 'Candidate pilot; outlet crosswalk and NWM routing agreement still require verification',
        'normalization': 'None', 'future_rainfall_policy': 'Historical AORC is not an operational future-rainfall input',
    })
    units = 'hours since 1970-01-01 00:00:00'
    encoding = {name: {'zlib': True, 'complevel': 2} for name in combined.data_vars if name != 'time_bounds'}
    encoding.update({'time': {'units': units, 'dtype': 'float64'}, 'time_bounds': {'units': units, 'dtype': 'float64'}, 'forcing_available': {'dtype': 'uint8'}})
    destination = output / f'{args.pilot}_{start:%Y%m%d}_{stop:%Y%m%d}_six_hour.nc'
    partial = destination.with_suffix('.nc.part')
    combined.to_netcdf(partial, engine='netcdf4', encoding=encoding)
    # Verify the serialized timeline, ordering and values before publishing the file.
    with xr.open_dataset(partial, engine='netcdf4') as check:
        if not pd.DatetimeIndex(check.time.values).equals(times) or not np.array_equal(check.feature_id, ids):
            raise ValueError('Output coordinates changed during serialization')
        for name in VARIABLES + ['streamflow']:
            if not np.array_equal(check[name].values, combined[name].values, equal_nan=True):
                raise ValueError(f'Output values changed: {name}')
    partial.replace(destination)
    # Issue index is the first future interval; history ends before issue time.
    issue_indices = np.arange(120, len(times)-40+1, dtype='int64')
    if not len(issue_indices):
        raise ValueError('Dataset too short for 120 history + 40 future intervals')
    np.savez_compressed(output / 'window_indices.npz', issue_index=issue_indices,
        issue_time=times.values[issue_indices], history_start=issue_indices-120,
        history_stop_exclusive=issue_indices, target_start=issue_indices,
        target_stop_exclusive=issue_indices+40, node_comid=ids)
    pd.DataFrame(audits).to_csv(output / 'coverage_audit.csv', index=False)
    report = {'pilot': args.pilot, 'nodes': len(ids), 'six_hour_intervals': len(times),
        'candidate_windows': len(issue_indices), 'history_steps': 120, 'horizon_steps': 40,
        'first_issue': str(times[issue_indices[0]]), 'last_issue': str(times[issue_indices[-1]]),
        'candidate_window_policy': 'Temporal windows only; target/input masks still applied by the future loader',
        'node_order_sha256': hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
        'aorc_grid_cells_read_per_hour': len(grid_columns),
        'forcing_nodes': int(available.sum()), 'output': str(destination),
        'training': 'Not run', 'sources': source_records}
    (output / 'dataset_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, indent=2))
    print(pd.DataFrame(audits).groupby('variable')[['valid_intervals', 'masked_intervals', 'partial_intervals']].sum().to_string())


if __name__ == '__main__':
    main()
