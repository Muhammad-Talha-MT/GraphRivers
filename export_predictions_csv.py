"""
Export GraphRivers predictions in the SAME long-format schema as Vinh's
predictions_long.csv.gz, so the two can be concatenated directly for
downstream comparison.

Columns (must match exactly):
  model, protocol, site_id, site_role, target_time_utc,
  target_hour_since_1970, lead_hours, observed_m3s, predicted_m3s

Notes on fields that don't map 1:1 from GraphRivers' setup:
- protocol: GraphRivers' in_dim includes antecedent NWM flow as a feature,
  which is neither Vinh's Test A (forcing-only) nor Test B (own observed Q)
  cleanly. Rather than mislabel it, this script tags GraphRivers rows with
  protocol='GraphRivers' (not A/B/C) -- flag this when merging with Vinh's
  table so nobody treats it as a strict Test-A-equivalent.
- lead_hours: GraphRivers predicts 1 day (24h) ahead only, so every row
  gets lead_hours=24 (no multi-lead-time sweep like Vinh's Test B).
- site_role: 'training_gauge' for all 12 (GraphRivers trains against the
  full 2,485-node network including all 12 gage nodes; there is no
  held-out/unseen-gauge zero-shot test in the current pipeline).

Usage:
  python export_predictions_csv.py --config config_frenchbroad.yaml \
      --checkpoint models/frenchbroad/gwnet2bestmodel_..._log.pth \
      --model-name "GraphRivers-Stage1" \
      --out outputs/frenchbroad/predictions_long_stage1.csv.gz

  python export_predictions_csv.py --config config_frenchbroad.yaml \
      --checkpoint models/frenchbroad/gwnetfinetune_bestmodel_..._log.pth \
      --model-name "GraphRivers-Stage2-finetuned" \
      --out outputs/frenchbroad/predictions_long_stage2.csv.gz
"""
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os, sys
import gzip
import csv
import time

from loadrivernet import loadNWMDF, formAdjacencyMat
from nwmutils import getUSGSData
from gwnetmodel import GWNet
from util_gtnet import load_adj, load_config
from trainwavenetwu_hourly import getDataLoaders, test
from finetune import extractGageData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(args, A, in_dim, out_dim=1):
    num_nodes = A.shape[0]
    seq = args.data.seq_length
    adj_mx = load_adj(A, args.gwnet_params.adjtype)
    supports = None if args.gwnet_params.aptonly else [torch.tensor(i).to(device) for i in adj_mx]

    if seq == 30:
        blocks, layers, kernel_size = 7, 2, 4
    elif seq in (60, 90, 120):
        blocks, layers, kernel_size = 4, 4, 4
    elif seq == 180:
        blocks, layers, kernel_size = 3, 4, 6
    elif seq == 365:
        blocks, layers, kernel_size = 4, 4, 8
    else:
        raise Exception('not implemented')

    model = GWNet(device, num_nodes=num_nodes, dropout=args.train_params.dropout,
        supports=supports, gcn_bool=args.gwnet_params.gcn_bool, addaptadj=args.gwnet_params.addaptadj,
        aptinit=None, in_dim=in_dim, out_dim=out_dim,
        residual_channels=args.gwnet_params.nhid, dilation_channels=args.gwnet_params.nhid,
        skip_channels=args.gwnet_params.nhid*4, end_channels=args.gwnet_params.nhid*8,
        apt_size=args.gwnet_params.apt_size, blocks=blocks, layers=layers,
        kernel_size=kernel_size, seq_len=seq)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
        help='Full path to the .pth checkpoint to export predictions from')
    parser.add_argument('--model-name', type=str, required=True,
        help='Value to put in the model column, e.g. GraphRivers-Stage1')
    parser.add_argument('--protocol', type=str, default='GraphRivers',
        help='Value to put in the protocol column (default: GraphRivers, since it does not map cleanly to A/B/C)')
    parser.add_argument('--out', type=str, required=True,
        help='Output path, e.g. outputs/frenchbroad/predictions_long_stage1.csv.gz')
    cargs = parser.parse_args()
    args = load_config(cargs.config)

    t0 = time.time()
    print("Loading NWM dataframe and network...")
    comIDset, comIDdict, allDF, goodInd, gageDict = loadNWMDF(args)

    print("Loading real USGS gage observations...")
    if args.data.interval == '3H':
        usgsDict = getUSGSData(args.watershed_name, gageDict.values(), reLoad=args.data.reload_usgs, valueType='iv')
    else:
        usgsDict = getUSGSData(args.watershed_name, gageDict.values(), reLoad=args.data.reload_usgs, valueType='dv')

    A = formAdjacencyMat(args.rootdir, args.watershed_name, goodInd, comIDdict,
        isWeighting=args.network.weighted_adj, isDirected=args.network.directed_adj,
        connect_exta_path=args.network.connect_exta_path, add_self_loop=True, nwm_ver=args.data.nwm_ver)
    nnode = A.shape[0]
    outputdir = os.path.join(args.rootdir, 'data/{0}'.format(args.watershed_name))

    if args.data.forcing_source not in ['era5', 'ERA5']:
        raise NotImplementedError("this script currently assumes ERA5 forcing_source")
    forcingDF = pkl.load(open('data/{0}/era5forcing_full_nwm{1}_{2}.pkl'.format(
        args.watershed_name, args.data.nwm_ver, args.data.interval), 'rb'))

    print("Building data loaders...")
    trainLoader, valLoader, testLoader, nfeatures = getDataLoaders(
        watershed=args.watershed_name, in_dim=args.gwnet_params.in_dim,
        batchsize=args.train_params.batch_size, seq=args.data.seq_length, comIDSet=comIDset,
        forcingDF=forcingDF, allDF=allDF, outputdir=outputdir,
        uselog=args.data.uselog, addstatics=args.data.addstatics,
        addnwm=args.data.addnwm, removeSWE=False)
    print(f'elapsed time {time.time()-t0:.1f}s')

    sample_comid = list(forcingDF.keys())[0]
    forcing_index = forcingDF[sample_comid].index
    common_start = max(allDF.index.min(), forcing_index.min())
    common_end = min(allDF.index.max(), forcing_index.max())
    allDF_trimmed = allDF.loc[common_start:common_end]
    del forcingDF
    trainLen = allDF_trimmed.index.searchsorted(pd.Timestamp('2010-12-31')) + 1
    valEnd = allDF_trimmed.index.searchsorted(pd.Timestamp('2015-12-31')) + 1
    valLen = valEnd - trainLen
    test_dates = allDF_trimmed.index[trainLen+valLen:]

    print(f"\nLoading checkpoint: {cargs.checkpoint}")
    if not os.path.exists(cargs.checkpoint):
        print(f"ERROR: checkpoint not found: {cargs.checkpoint}")
        sys.exit(1)
    model = build_model(args, A, nfeatures, out_dim=1).to(device)
    model.load_state_dict(torch.load(cargs.checkpoint, map_location=device))
    model.eval()

    print("Running inference on test set...")
    _, predMat = test(args, model, testLoader, nnode, invTransform=True, reGen=True)
    assert len(test_dates) == predMat.shape[0], \
        f"date/row mismatch: {len(test_dates)} dates vs {predMat.shape[0]} pred rows"

    usgsMatO, colIDs = extractGageData(usgsDict, gageDict, comIDdict)
    usgsMatO_test = usgsMatO.reindex(test_dates)

    print(f"Writing {cargs.out} ...")
    os.makedirs(os.path.dirname(cargs.out), exist_ok=True)
    n_rows = 0
    with gzip.open(cargs.out, 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'protocol', 'site_id', 'site_role', 'target_time_utc',
                          'target_hour_since_1970', 'lead_hours', 'observed_m3s', 'predicted_m3s'])
        for item, col in zip(gageDict.keys(), colIDs):
            usgsid = gageDict[item]
            obs_series = usgsMatO_test[col]
            for i, date in enumerate(test_dates):
                obs_val = obs_series.iloc[i]
                pred_val = predMat[i, col]
                target_time_utc = date.strftime('%Y-%m-%dT%H:%M:%SZ')
                target_hour_since_1970 = int(date.value // 10**9 // 3600)
                writer.writerow([
                    cargs.model_name,
                    cargs.protocol,
                    usgsid,
                    'training_gauge',
                    target_time_utc,
                    target_hour_since_1970,
                    24,
                    '' if pd.isna(obs_val) else float(obs_val),
                    float(pred_val),
                ])
                n_rows += 1

    print(f"\nWrote {n_rows} rows to {cargs.out}")
    print(f"({len(test_dates)} test dates x {len(colIDs)} gages)")


if __name__ == '__main__':
    main()
