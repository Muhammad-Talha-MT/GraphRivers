"""
Diagnostic: verify that USGS gage observations align correctly with the
trimmed allDF date range used by the training/val/test loaders, BEFORE
running finetune.py for real. finetune.py assumes positional alignment
(usgsMatO row i corresponds to allDF row i) with no explicit date check.

Usage:
  python check_finetune_alignment.py --config config_frenchbroad.yaml
"""
import argparse
import numpy as np
import pandas as pd

from loadrivernet import loadNWMDF, formAdjacencyMat
from genrivernet import loadWatershed
from nwmutils import getUSGSData
from util_gtnet import load_config
from finetune import extractGageData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cargs = parser.parse_args()
    args = load_config(cargs.config)

    print("=" * 70)
    print("STEP 1: Load allDF (NWM data) exactly as loadNWMDF does")
    print("=" * 70)
    comIDset, comIDdict, allDF, goodInd, gageDict = loadNWMDF(args)
    print(f"allDF shape: {allDF.shape}")
    print(f"allDF date range: {allDF.index.min()} to {allDF.index.max()}")
    print(f"allDF index dtype: {allDF.index.dtype}")

    print()
    print("=" * 70)
    print("STEP 2: Load USGS gage data exactly as finetune.py does")
    print("=" * 70)
    if args.data.interval == '3H':
        usgsDict = getUSGSData(args.watershed_name, gageDict.values(), reLoad=args.data.reload_usgs, valueType='iv')
    else:
        usgsDict = getUSGSData(args.watershed_name, gageDict.values(), reLoad=args.data.reload_usgs, valueType='dv')

    usgsMatO, colIDs = extractGageData(usgsDict, gageDict, comIDdict)
    print(f"usgsMatO shape: {usgsMatO.shape}")
    print(f"usgsMatO date range: {usgsMatO.index.min()} to {usgsMatO.index.max()}")
    print(f"usgsMatO index dtype: {usgsMatO.index.dtype}")
    print(f"colIDs (comID -> column positions in allDF/model output): {colIDs}")

    print()
    print("=" * 70)
    print("STEP 3: Reproduce the SAME date-alignment trim that")
    print("genMLDataSetsWithForcing applies to allDF (forcing-vs-NWM overlap)")
    print("=" * 70)
    import pickle as pkl
    forcingDF = pkl.load(open('data/{0}/era5forcing_full_nwm{1}_{2}.pkl'.format(
        args.watershed_name, args.data.nwm_ver, args.data.interval), 'rb'))
    sample_comid = comIDset[0] if hasattr(comIDset, '__getitem__') else list(comIDset)[0]
    forcing_index = forcingDF[sample_comid].index
    common_start = max(allDF.index.min(), forcing_index.min())
    common_end = min(allDF.index.max(), forcing_index.max())
    allDF_trimmed = allDF.loc[common_start:common_end]
    print(f"Trimmed allDF (what the model ACTUALLY trains on) range: {common_start} to {common_end}")
    print(f"Trimmed allDF shape: {allDF_trimmed.shape}")

    print()
    print("=" * 70)
    print("STEP 4: THE CRITICAL CHECK")
    print("=" * 70)
    print(f"Does usgsMatO's date range match the TRIMMED allDF range?")
    print(f"  usgsMatO starts:      {usgsMatO.index.min()}")
    print(f"  trimmed allDF starts: {common_start}")
    print(f"  MATCH: {usgsMatO.index.min() == common_start}")
    print()
    print(f"  usgsMatO ends:      {usgsMatO.index.max()}")
    print(f"  trimmed allDF ends: {common_end}")
    print(f"  MATCH: {usgsMatO.index.max() == common_end}")
    print()
    print(f"  usgsMatO row count:      {usgsMatO.shape[0]}")
    print(f"  trimmed allDF row count: {allDF_trimmed.shape[0]}")
    print(f"  MATCH: {usgsMatO.shape[0] == allDF_trimmed.shape[0]}")

    print()
    if (usgsMatO.index.min() == common_start and
        usgsMatO.index.max() == common_end and
        usgsMatO.shape[0] == allDF_trimmed.shape[0]):
        print("SAFE: usgsMatO aligns exactly with the trimmed allDF used by the loaders.")
        print("  finetune.py's positional assumption (row i <-> row i) HOLDS.")
    else:
        print("MISALIGNED: finetune.py's positional row-alignment assumption is BROKEN.")
        print("  Running finetune.py as-is would silently pair wrong dates' observations")
        print("  with wrong dates' training samples. DO NOT run finetune.py until this")
        print("  is fixed.")

    print()
    print("=" * 70)
    print("STEP 5: Spot-check overlapping dates directly (belt and suspenders)")
    print("=" * 70)
    check_dates = [common_start, common_start + pd.Timedelta(days=100),
                   common_end - pd.Timedelta(days=100), common_end]
    for d in check_dates:
        in_usgs = d in usgsMatO.index
        in_trimmed = d in allDF_trimmed.index
        print(f"  Date {d}: in usgsMatO={in_usgs}, in trimmed allDF={in_trimmed}")


if __name__ == '__main__':
    main()
