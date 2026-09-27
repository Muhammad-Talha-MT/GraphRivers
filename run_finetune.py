"""
Stage II: fine-tune a Stage I checkpoint against real USGS observations,
then evaluate the result against real USGS data (not NWM) for genuine
comparability with external baselines.

Usage:
  python run_finetune.py --config config_frenchbroad.yaml
"""
import random
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os, sys
import json
import time

from loadrivernet import loadNWMDF, formAdjacencyMat
from genrivernet import loadWatershed
from nwmutils import getUSGSData
from gwnetmodel import GWNet
from util_gtnet import load_adj, load_config
from trainwavenetwu_hourly import getDataLoaders, test
from finetune import finetune, extractGageData

import hydrostats as Hydrostats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model_and_load_stage1_checkpoint(args, A, in_dim, save_path, out_dim=1):
    num_nodes = A.shape[0]
    seq = args.data.seq_length
    addaptadj = args.gwnet_params.addaptadj

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
        supports=supports, gcn_bool=args.gwnet_params.gcn_bool, addaptadj=addaptadj,
        aptinit=None, in_dim=in_dim, out_dim=out_dim,
        residual_channels=args.gwnet_params.nhid, dilation_channels=args.gwnet_params.nhid,
        skip_channels=args.gwnet_params.nhid*4, end_channels=args.gwnet_params.nhid*8,
        apt_size=args.gwnet_params.apt_size, blocks=blocks, layers=layers,
        kernel_size=kernel_size, seq_len=seq)

    seed = args.seed
    lossfunstr = 'L1' if args.train_params.L1Loss else 'L2'
    if addaptadj:
        basestr = f'seq{seq}_{lossfunstr}_seed{seed}_node{num_nodes}_aptadj_nwm{args.data.nwm_ver}_{args.data.forcing_source}_{args.data.interval}'
    else:
        basestr = f'seq{seq}_{lossfunstr}_seed{seed}_node{num_nodes}_nwm{args.data.nwm_ver}_{args.data.forcing_source}_{args.data.interval}'
    if args.data.addnwm: basestr += '_usenwm'
    if args.data.uselabel: basestr += '_label'
    if args.network.directed_adj: basestr += '_directed'
    if args.network.weighted_adj: basestr += '_weighted'
    if args.network.connect_exta_path: basestr += '_extrapath'

    suffix = '_log' if args.data.uselog else ''
    model_path = '/'.join([save_path, f'gwnet2bestmodel_{basestr}{suffix}.pth'])

    if not os.path.exists(model_path):
        print(f"ERROR: Stage I checkpoint not found at {model_path}")
        sys.exit(1)

    print(f"Loading Stage I checkpoint: {model_path}")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, basestr


def compute_nse_vs_real_usgs(predMat, allDF_trimmed_index, trainLen, valLen,
                              usgsDict, gageDict, comIDdict, label):
    """predMat: (n_test_days, n_nodes) array in real flow units, already
    inverse-transformed. Compares only at the gaged columns, against REAL
    USGS observations over the exact same test-period dates.
    """
    test_dates = allDF_trimmed_index[trainLen+valLen:]
    assert len(test_dates) == predMat.shape[0], \
        f"date/row mismatch: {len(test_dates)} dates vs {predMat.shape[0]} pred rows"

    usgsMatO, colIDs = extractGageData(usgsDict, gageDict, comIDdict)
    usgsMatO_test = usgsMatO.reindex(test_dates)

    results = []
    for item, col in zip(gageDict.keys(), colIDs):
        usgsid = gageDict[item]
        obs = usgsMatO_test[col].values if col in usgsMatO_test.columns else usgsMatO_test.iloc[:, list(colIDs).index(col)].values
        sim = predMat[:, col]
        df = pd.DataFrame({'sim': sim, 'obs': obs}).dropna()
        if df.shape[0] < 10:
            print(f"  WARNING: only {df.shape[0]} valid obs for gage {usgsid}, skipping")
            continue
        nse = Hydrostats.nse(df['sim'], df['obs'])
        results.append({'usgsid': usgsid, 'comid': item, 'n': df.shape[0], 'nse': nse})

    print(f"\n--- {label}: NSE vs REAL USGS observations, per gage ---")
    for r in results:
        print(f"  {r['usgsid']} (comid {r['comid']}): NSE={r['nse']:.4f}  (n={r['n']})")
    nse_vals = [r['nse'] for r in results]
    print(f"  MEDIAN NSE: {np.median(nse_vals):.4f}   MEAN NSE: {np.mean(nse_vals):.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cargs = parser.parse_args()
    args = load_config(cargs.config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    save_path = os.path.join(args.rootdir, 'models/{0}'.format(args.watershed_name))
    seq = args.data.seq_length
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

    if args.data.forcing_source in ['era5', 'ERA5']:
        forcingDF = pkl.load(open('data/{0}/era5forcing_full_nwm{1}_{2}.pkl'.format(
            args.watershed_name, args.data.nwm_ver, args.data.interval), 'rb'))
    else:
        raise NotImplementedError("this script currently assumes ERA5 forcing_source")

    print("Building data loaders...")
    trainLoader, valLoader, testLoader, nfeatures = getDataLoaders(
        watershed=args.watershed_name, in_dim=args.gwnet_params.in_dim,
        batchsize=args.train_params.batch_size, seq=seq, comIDSet=comIDset,
        forcingDF=forcingDF, allDF=allDF, outputdir=outputdir,
        uselog=args.data.uselog, addstatics=args.data.addstatics,
        addnwm=args.data.addnwm, removeSWE=False)
    print(f'elapsed time {time.time()-t0:.1f}s')

    # recompute the exact trimmed date index + split, same as finetune()'s internal logic,
    # so we can align predictions to real dates for the final USGS comparison
    sample_comid = list(forcingDF.keys())[0]
    forcing_index = forcingDF[sample_comid].index
    common_start = max(allDF.index.min(), forcing_index.min())
    common_end = min(allDF.index.max(), forcing_index.max())
    allDF_trimmed = allDF.loc[common_start:common_end]
    del forcingDF
    trainLen = allDF_trimmed.index.searchsorted(pd.Timestamp('2010-12-31')) + 1
    valEnd = allDF_trimmed.index.searchsorted(pd.Timestamp('2015-12-31')) + 1
    valLen = valEnd - trainLen

    print("\nLoading Stage I model...")
    model, basestr = build_model_and_load_stage1_checkpoint(args, A, nfeatures, save_path=save_path, out_dim=1)

    print("\n" + "="*70)
    print("BASELINE: Stage I model (no fine-tuning) vs REAL USGS")
    print("="*70)
    trueMat_nwm, predMat_stage1 = test(args, model, testLoader, nnode, invTransform=True, reGen=True)
    compute_nse_vs_real_usgs(predMat_stage1, allDF_trimmed.index, trainLen, valLen,
                              usgsDict, gageDict, comIDdict, label="Stage I (pretrained only)")
    print("\n(for reference) NWM itself vs REAL USGS:")
    compute_nse_vs_real_usgs(trueMat_nwm, allDF_trimmed.index, trainLen, valLen,
                              usgsDict, gageDict, comIDdict, label="NWM simulated flow")

    print("\n" + "="*70)
    print("RUNNING STAGE II: fine-tuning against real USGS observations")
    print("="*70)
    model = finetune(args, model, "gwnetfinetune", trainLoader, valLoader, A,
        save_path, usgsDict, gageDict, comIDdict, basestr, reTrain=True)

    print("\n" + "="*70)
    print("AFTER FINE-TUNING: model vs REAL USGS")
    print("="*70)
    _, predMat_finetuned = test(args, model, testLoader, nnode, invTransform=True, reGen=True)
    results_ft = compute_nse_vs_real_usgs(predMat_finetuned, allDF_trimmed.index, trainLen, valLen,
                              usgsDict, gageDict, comIDdict, label="Stage II (fine-tuned)")

    report = {
        "watershed": args.watershed_name,
        "test_period": f"{allDF_trimmed.index[trainLen+valLen]} to {allDF_trimmed.index[-1]}",
        "fine_tune_epochs": args.fine_tune.nepoch,
        "fine_tune_lr": args.fine_tune.learnrate,
        "results_vs_real_usgs": results_ft,
        "median_nse": float(np.median([r['nse'] for r in results_ft])),
    }
    outpath = f"outputs/{args.watershed_name}/finetune_report.json"
    os.makedirs(f"outputs/{args.watershed_name}", exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved report to {outpath}")


if __name__ == '__main__':
    main()
