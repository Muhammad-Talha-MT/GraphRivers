"""
Standalone evaluation script for GraphRivers French Broad model.
Loads an EXISTING checkpoint (does not train) and runs test + NSE evaluation.

Usage:
  python eval_checkpoint.py --config config_frenchbroad.yaml
"""
import random
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os, sys
import time

from loadrivernet import loadNWMDF, formAdjacencyMat, genMLDataSetsWithForcing
from genrivernet import loadWatershed
from nwmutils import plotNetworkNSE, getUSGSData, compareNWMUSGS
from gwnetmodel import GWNet
from util_gtnet import Optim, load_adj, load_config

from trainwavenetwu_hourly import getDataLoaders, test, getNSE, plotNSE, doCompareNWM_ML_USGS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model_and_load_checkpoint(args, A, in_dim, save_path, out_dim=1):
    num_nodes = A.shape[0]
    seq = args.data.seq_length
    addaptadj = args.gwnet_params.addaptadj

    adj_mx = load_adj(A, args.gwnet_params.adjtype)
    if args.gwnet_params.aptonly:
        supports = None
    else:
        supports = [torch.tensor(i).to(device) for i in adj_mx]

    if seq == 30:
        blocks, layers, kernel_size = 7, 2, 4
    elif seq == 60:
        blocks, layers, kernel_size = 4, 4, 4
    elif seq == 90:
        blocks, layers, kernel_size = 4, 4, 4
    elif seq == 120:
        blocks, layers, kernel_size = 4, 4, 4
    elif seq == 180:
        blocks, layers, kernel_size = 3, 4, 6
    elif seq == 365:
        blocks, layers, kernel_size = 4, 4, 8
    else:
        raise Exception('not implemented')

    model = GWNet(device,
        num_nodes=num_nodes,
        dropout=args.train_params.dropout,
        supports=supports,
        gcn_bool=args.gwnet_params.gcn_bool,
        addaptadj=addaptadj,
        aptinit=None,
        in_dim=in_dim,
        out_dim=out_dim,
        residual_channels=args.gwnet_params.nhid,
        dilation_channels=args.gwnet_params.nhid,
        skip_channels=args.gwnet_params.nhid * 4,
        end_channels=args.gwnet_params.nhid * 8,
        apt_size=args.gwnet_params.apt_size,
        blocks=blocks,
        layers=layers,
        kernel_size=kernel_size, seq_len=seq)

    seed = args.seed
    lossfunstr = 'L1' if args.train_params.L1Loss else 'L2'
    if addaptadj:
        basestr = f'seq{seq}_{lossfunstr}_seed{seed}_node{num_nodes}_aptadj_nwm{args.data.nwm_ver}_{args.data.forcing_source}_{args.data.interval}'
    else:
        basestr = f'seq{seq}_{lossfunstr}_seed{seed}_node{num_nodes}_nwm{args.data.nwm_ver}_{args.data.forcing_source}_{args.data.interval}'
    if args.data.addnwm:
        basestr += '_usenwm'
    if args.data.uselabel:
        basestr += '_label'
    if args.network.directed_adj:
        basestr += '_directed'
    if args.network.weighted_adj:
        basestr += '_weighted'
    if args.network.connect_exta_path:
        basestr += '_extrapath'

    if args.data.uselog:
        model_path = '/'.join([save_path, f'gwnet2bestmodel_{basestr}_log.pth'])
    else:
        model_path = '/'.join([save_path, f'gwnet2bestmodel_{basestr}.pth'])

    if not os.path.exists(model_path):
        print(f"ERROR: checkpoint not found at {model_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {model_path}")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


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

    if args.data.interval == '3H':
        usgsDict = getUSGSData(args.watershed_name, gageDict.values(), reLoad=args.data.reload_usgs, valueType='iv')
    else:
        usgsDict = getUSGSData(args.watershed_name, gageDict.values(), reLoad=args.data.reload_usgs, valueType='dv')

    A = formAdjacencyMat(args.rootdir, args.watershed_name, goodInd, comIDdict,
        isWeighting=args.network.weighted_adj, isDirected=args.network.directed_adj,
        connect_exta_path=args.network.connect_exta_path, add_self_loop=True, nwm_ver=args.data.nwm_ver)

    nnode = A.shape[0]
    outputdir = os.path.join(args.rootdir, 'data/{0}'.format(args.watershed_name))

    if args.data.forcing_source in ['daymet', 'Daymet']:
        forcingDF = pkl.load(open('data/{0}/daymetforcing{1}_nwm{2}.pkl'.format(args.watershed_name, nnode, args.data.nwm_ver), 'rb'))
        removeSWE = args.data.removeSWE
    elif args.data.forcing_source in ['nldas', 'NLDAS']:
        forcingDF = pkl.load(open('data/{0}/nldasforcing{1}_nwm{2}_{3}.pkl'.format(args.watershed_name, nnode, args.data.nwm_ver, args.data.interval), 'rb'))
        removeSWE = False
    elif args.data.forcing_source in ['aorc', 'AORC']:
        forcingDF = pkl.load(open('data/{0}/aorcforcing{1}_nwm{2}_{3}.pkl'.format(args.watershed_name, nnode, args.data.nwm_ver, args.data.interval), 'rb'))
        removeSWE = False
    elif args.data.forcing_source in ['era5', 'ERA5']:
        forcingDF = pkl.load(open('data/{0}/era5forcing_full_nwm{1}_{2}.pkl'.format(args.watershed_name, args.data.nwm_ver, args.data.interval), 'rb'))
        removeSWE = False
    else:
        raise ValueError("invalid forcing source")

    print("Building data loaders (test set)...")
    trainLoader, valLoader, testLoader, nfeatures = getDataLoaders(
        watershed=args.watershed_name,
        in_dim=args.gwnet_params.in_dim,
        batchsize=args.train_params.batch_size,
        seq=seq,
        comIDSet=comIDset,
        forcingDF=forcingDF,
        allDF=allDF,
        outputdir=outputdir,
        uselog=args.data.uselog,
        addstatics=args.data.addstatics,
        addnwm=args.data.addnwm,
        removeSWE=removeSWE
    )
    print(f'elapsed time {time.time()-t0:.1f}s')

    print("Loading model checkpoint (no training)...")
    model = build_model_and_load_checkpoint(args, A, nfeatures, save_path=save_path, out_dim=1)

    print("Running inference on test set...")
    trueMat, outMat = test(args, model, testLoader, nnode, invTransform=True, reGen=True)

    print("Computing NSE...")
    nsevecF = getNSE(trueMat, outMat)

    print("Generating NSE distribution plot (no shapefile needed)...")
    try:
        import matplotlib.pyplot as plt
        os.makedirs(f"outputs/{args.watershed_name}", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        axes[0].hist(nsevecF, bins=50, color="steelblue", edgecolor="black")
        axes[0].set_xlabel("NSE"); axes[0].set_ylabel("Number of nodes")
        axes[0].set_title(f"NSE histogram (n={len(nsevecF)} nodes)")
        sorted_nse = np.sort(nsevecF)
        ecdf_y = np.arange(1, len(sorted_nse)+1) / len(sorted_nse)
        axes[1].plot(sorted_nse, ecdf_y, color="darkorange")
        axes[1].set_xlabel("NSE"); axes[1].set_ylabel("ECDF")
        axes[1].set_title("Empirical CDF of NSE")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        outpath = f"outputs/{args.watershed_name}/nse_distribution_checkpoint_eval.png"
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"Saved NSE distribution plot to {outpath}")
    except Exception as e:
        print(f"NSE distribution plot failed (non-fatal): {e}")

    print("Comparing NWM vs ML vs USGS at gage locations...")
    try:
        doCompareNWM_ML_USGS(args, comIDdict, gageDict, usgsDict, nwmMat=trueMat, mlMat=outMat,
            allDF=allDF, modelName="gwmodel_checkpoint_eval", interval=args.data.interval, usgsvar='iv')
    except Exception as e:
        print(f"doCompareNWM_ML_USGS failed (non-fatal): {e}")

    print("Done. NSE summary saved above; check outputs/{0}/ for plots.".format(args.watershed_name))


if __name__ == '__main__':
    main()
