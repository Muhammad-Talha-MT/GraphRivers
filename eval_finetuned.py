"""
Evaluate a fine-tuned (Stage II) checkpoint against real USGS observations,
in a format directly comparable to external baselines (e.g. labmate's
HydroORBIT/LSTM tables): per-gage NSE and KGE, plus median/mean.

Usage:
  python eval_finetuned.py --config config_frenchbroad.yaml
"""
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os, sys
import json
import time

from loadrivernet import loadNWMDF, formAdjacencyMat
from nwmutils import getUSGSData
from gwnetmodel import GWNet
from util_gtnet import load_adj, load_config
from trainwavenetwu_hourly import getDataLoaders, test
from finetune import extractGageData
import hydrostats as Hydrostats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_basestr(args, num_nodes):
    seed = args.seed
    lossfunstr = 'L1' if args.train_params.L1Loss else 'L2'
    if args.gwnet_params.addaptadj:
        basestr = f'seq{args.data.seq_length}_{lossfunstr}_seed{seed}_node{num_nodes}_aptadj_nwm{args.data.nwm_ver}_{args.data.forcing_source}_{args.data.interval}'
    else:
        basestr = f'seq{args.data.seq_length}_{lossfunstr}_seed{seed}_node{num_nodes}_nwm{args.data.nwm_ver}_{args.data.forcing_source}_{args.data.interval}'
    if args.data.addnwm: basestr += '_usenwm'
    if args.data.uselabel: basestr += '_label'
    if args.network.directed_adj: basestr += '_directed'
    if args.network.weighted_adj: basestr += '_weighted'
    if args.network.connect_exta_path: basestr += '_extrapath'
    return basestr


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


def evaluate_checkpoint_vs_usgs(model_path, model, testLoader, args, nnode,
                                 allDF_trimmed_index, trainLen, valLen,
                                 usgsDict, gageDict, comIDdict, label):
    if not os.path.exists(model_path):
        print(f"ERROR: checkpoint not found: {model_path}")
        sys.exit(1)
    print(f"\nLoading checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Running inference on test set ({label})...")
    _, predMat = test(args, model, testLoader, nnode, invTransform=True, reGen=True)

    test_dates = allDF_trimmed_index[trainLen+valLen:]
    assert len(test_dates) == predMat.shape[0], \
        f"date/row mismatch: {len(test_dates)} dates vs {predMat.shape[0]} pred rows"

    usgsMatO, colIDs = extractGageData(usgsDict, gageDict, comIDdict)
    usgsMatO_test = usgsMatO.reindex(test_dates)

    results = []
    for item, col in zip(gageDict.keys(), colIDs):
        usgsid = gageDict[item]
        obs = usgsMatO_test[col].values
        sim = predMat[:, col]
        df = pd.DataFrame({'sim': sim, 'obs': obs}).dropna()
        if df.shape[0] < 10:
            print(f"  WARNING: only {df.shape[0]} valid obs for gage {usgsid}, skipping")
            continue
        nse = Hydrostats.nse(df['sim'], df['obs'])
        try:
            kge = Hydrostats.kge_2009(df['sim'], df['obs'])
        except Exception:
            kge = float('nan')
        results.append({'usgsid': usgsid, 'comid': item, 'n': int(df.shape[0]),
                         'nse': float(nse), 'kge': float(kge)})

    print(f"\n--- {label}: NSE/KGE vs REAL USGS observations, per gage ---")
    print(f"{'usgsid':<14}{'comid':<12}{'n':<8}{'NSE':<10}{'KGE':<10}")
    for r in results:
        print(f"{r['usgsid']:<14}{r['comid']:<12}{r['n']:<8}{r['nse']:<10.4f}{r['kge']:<10.4f}")
    nse_vals = [r['nse'] for r in results]
    kge_vals = [r['kge'] for r in results]
    print(f"\nMEDIAN NSE: {np.median(nse_vals):.4f}   MEAN NSE: {np.mean(nse_vals):.4f}")
    print(f"MEDIAN KGE: {np.median(kge_vals):.4f}   MEAN KGE: {np.mean(kge_vals):.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None,
        help='Override: full path to a specific checkpoint to evaluate. '
             'Default: the fine-tuned (gwnetfinetune) best model.')
    parser.add_argument('--compare_stage1', action='store_true',
        help='Also evaluate the Stage I (pretrained-only) checkpoint for side-by-side comparison.')
    cargs = parser.parse_args()
    args = load_config(cargs.config)

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

    if args.data.forcing_source not in ['era5', 'ERA5']:
        raise NotImplementedError("this script currently assumes ERA5 forcing_source")
    forcingDF = pkl.load(open('data/{0}/era5forcing_full_nwm{1}_{2}.pkl'.format(
        args.watershed_name, args.data.nwm_ver, args.data.interval), 'rb'))

    print("Building data loaders...")
    trainLoader, valLoader, testLoader, nfeatures = getDataLoaders(
        watershed=args.watershed_name, in_dim=args.gwnet_params.in_dim,
        batchsize=args.train_params.batch_size, seq=seq, comIDSet=comIDset,
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

    basestr = build_basestr(args, nnode)
    suffix = '_log' if args.data.uselog else ''

    all_reports = {}

    if cargs.compare_stage1:
        stage1_path = cargs.checkpoint or '/'.join([save_path, f'gwnet2bestmodel_{basestr}{suffix}.pth'])
        model = build_model(args, A, nfeatures, out_dim=1).to(device)
        r1 = evaluate_checkpoint_vs_usgs(stage1_path, model, testLoader, args, nnode,
            allDF_trimmed.index, trainLen, valLen, usgsDict, gageDict, comIDdict,
            label="Stage I (pretrained only)")
        all_reports['stage1'] = r1

    finetuned_path = cargs.checkpoint or '/'.join([save_path, f'gwnetfinetune_bestmodel_{basestr}{suffix}.pth'])
    model = build_model(args, A, nfeatures, out_dim=1).to(device)
    r2 = evaluate_checkpoint_vs_usgs(finetuned_path, model, testLoader, args, nnode,
        allDF_trimmed.index, trainLen, valLen, usgsDict, gageDict, comIDdict,
        label="Stage II (fine-tuned)")
    all_reports['finetuned'] = r2

    outpath = f"outputs/{args.watershed_name}/eval_finetuned_report.json"
    os.makedirs(f"outputs/{args.watershed_name}", exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump({
            'watershed': args.watershed_name,
            'checkpoint_evaluated': finetuned_path,
            'reports': all_reports,
            'median_nse_finetuned': float(np.median([r['nse'] for r in r2])),
            'median_kge_finetuned': float(np.median([r['kge'] for r in r2])),
        }, f, indent=2, default=str)
    print(f"\nSaved report to {outpath}")


if __name__ == '__main__':
    main()
