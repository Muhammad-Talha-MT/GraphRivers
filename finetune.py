#author: alex sun
#date: 09242022
#purpose: start from a pre-trained model, incorporate observations
#-----------------------------------------------------------------
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import time
import tqdm
import os,sys
from util_gtnet import Optim 
import pickle as pkl
import hydrostats as Hydrostats
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extractGageData(usgsDict,gageDict,comIDdict):
    """extract usgs gage data
    Params:
    ------
    usgsDict, usgsid:Q mapping 3H
    gageDict, comid:usgsid
    comIDdict: comid:col no in the output tensor

    Returns:
    -------
    qmat, streamflow data
    colIDs, column ID
    """
    #FIX: build an explicit, controlled daily date range and reindex every
    #gage onto it directly, rather than relying on pd.concat's automatic
    #outer-join index union. Some pandas versions (observed: 3.0.6 with
    #datetime64[us, UTC] indices) can silently fabricate spurious out-of-range
    #index entries during outer-join concat -- explicit reindexing avoids this
    #entirely since we fully control the target index ourselves.
    colIDs = []
    series_list = []
    global_min, global_max = None, None
    for item in gageDict.keys():
        usgsid = gageDict[item]
        dfQ = usgsDict[usgsid]['Q'].copy()
        dfQ.index = pd.to_datetime(dfQ.index, utc=True).tz_localize(None)
        dfQ.name = comIDdict[item]  # FIX: name each series by its column ID so concat produces uniquely-named columns, not 12 duplicate 'Q' columns
        print(f'comid {item}, usgsid {usgsid}, col id {comIDdict[item]}, raw records {dfQ.shape[0]}')
        colIDs.append(comIDdict[item])
        series_list.append((item, usgsid, dfQ))
        gmin, gmax = dfQ.index.min(), dfQ.index.max()
        global_min = gmin if global_min is None else min(global_min, gmin)
        global_max = gmax if global_max is None else max(global_max, gmax)

    canonical_index = pd.date_range(start=global_min, end=global_max, freq='D')
    print(f'Canonical USGS index: {canonical_index.min()} to {canonical_index.max()} ({len(canonical_index)} days)')

    reindexed = []
    for item, usgsid, dfQ in series_list:
        dfQ_reindexed = dfQ.reindex(canonical_index)
        reindexed.append(dfQ_reindexed)

    allDF = pd.concat(reindexed, axis=1, join='outer')
    assert allDF.index.max() == global_max, f'index corruption still present: max={allDF.index.max()}, expected={global_max}'
    assert allDF.index.min() == global_min, f'index corruption still present: min={allDF.index.min()}, expected={global_min}'
    allDF.ffill(inplace=True)
    allDF.bfill(inplace=True)
    print ('allDF shape', allDF.shape)
    print ('nan values remaining', np.where(np.isnan(allDF.values))[0].shape[0])
    return allDF,colIDs

def disaggregateDaily2Hourly(allUSGSQDF, forcingDF, colIDs, freq=3):
    arr = []
    N = 24/freq
    nrec = forcingDF.shape[0]
    for i in range(0, nrec-N, N):
        df = forcingDF.iloc[i*N:(i+1)*N, colIDs]
        print (df)

def finetune(args,model,model_prefix,trainLoader,valLoader,A,save_path,usgsDict, gageDict,comIDdict, basestr, reTrain=False, **kwargs):
    """Main code for fine tuning
    """
    from loadrivernet import loadNWMDF

    usgsMatO,colIDs = extractGageData(usgsDict,gageDict,comIDdict)

    num_nodes = A.shape[0]
    nEpoch = args.fine_tune.nepoch
    lr = args.fine_tune.learnrate #this needs to be a small learning rate
    seq = args.data.seq_length
    usefinal = args.train_params.get("usefinal", False)
    uselog = args.data.uselog
    noffset = seq

    #FIX: independently rebuild the exact same trimmed date range and
    #train/val split that genMLDataSetsWithForcing used, so usgsMatO can
    #be aligned by DATE rather than assumed to align by ROW POSITION.
    #NOTE: these two date strings MUST match the trainEndDate/valEndDate
    #arguments passed to getDataLoaders() in trainwavenetwu_hourly.py.
    TRAIN_END_DATE = '2010-12-31'
    VAL_END_DATE = '2015-12-31'

    print('Recomputing trimmed date range for fine-tune alignment...')
    _, _, allDF_ft, _, _ = loadNWMDF(args)
    if args.data.forcing_source in ['era5','ERA5']:
        forcingDF_ft = pkl.load(open('data/{0}/era5forcing_full_nwm{1}_{2}.pkl'.format(
            args.watershed_name, args.data.nwm_ver, args.data.interval), 'rb'))
    else:
        raise NotImplementedError('finetune date-alignment currently only implemented for ERA5 forcing_source')
    sample_comid_ft = list(forcingDF_ft.keys())[0]
    forcing_index_ft = forcingDF_ft[sample_comid_ft].index
    common_start_ft = max(allDF_ft.index.min(), forcing_index_ft.min())
    common_end_ft = min(allDF_ft.index.max(), forcing_index_ft.max())
    allDF_trimmed_ft = allDF_ft.loc[common_start_ft:common_end_ft]
    del forcingDF_ft  # free ~1.5GB promptly, only needed the index

    trainLen = allDF_trimmed_ft.index.searchsorted(pd.Timestamp(TRAIN_END_DATE)) + 1
    valEnd   = allDF_trimmed_ft.index.searchsorted(pd.Timestamp(VAL_END_DATE)) + 1
    valLen   = valEnd - trainLen
    print(f'Fine-tune alignment: trimmed range {common_start_ft} to {common_end_ft} '
          f'({allDF_trimmed_ft.shape[0]} rows), trainLen={trainLen}, valLen={valLen}')

    n_train = len(trainLoader.dataset)
    n_val = len(valLoader.dataset)
    assert n_train == trainLen - noffset, \
        f'train split mismatch: loader has {n_train} samples, expected {trainLen-noffset}. ' \
        f'TRAIN_END_DATE/VAL_END_DATE in finetune.py may be out of sync with trainwavenetwu_hourly.py.'
    assert n_val == valLen, \
        f'val split mismatch: loader has {n_val} samples, expected {valLen}. ' \
        f'TRAIN_END_DATE/VAL_END_DATE in finetune.py may be out of sync with trainwavenetwu_hourly.py.'

    #reindex USGS observations onto the exact trimmed date grid.
    #dates with no real observation (e.g. before any gage's period of record)
    #become NaN here -- deliberately NOT filled, so the loss can mask them out
    #rather than fine-tuning against fabricated values.
    usgsMatO_aligned = usgsMatO.reindex(allDF_trimmed_ft.index)
    n_valid_total = usgsMatO_aligned.notna().sum().sum()
    n_total_cells = usgsMatO_aligned.shape[0] * usgsMatO_aligned.shape[1]
    print(f'USGS coverage over trimmed range: {n_valid_total}/{n_total_cells} cells have real observations '
          f'({100*n_valid_total/n_total_cells:.1f}%)')

    #need to transform the data
    if uselog:
        myscaler = pkl.load(open(f'data/{args.watershed_name}/myscaler_log.pkl', 'rb'))
    else:
        myscaler = pkl.load(open(f'data/{args.watershed_name}/myscaler.pkl', 'rb'))

    #FIX: this pipeline's scaler is a simple elementwise log/z-score transform
    #(myscaler has input_means/stds, output_mean/std, q_scale -- NOT a fitted
    #sklearn PowerTransformer). Apply the exact same per-column formula
    #getStandardScaler() uses for streamflow, directly to the USGS values.
    #NaN passes through both formulas untouched (log(nan)=nan, (nan-x)/y=nan),
    #so masking in getLoss() still works correctly downstream.
    usgs_vals = usgsMatO_aligned.values  # (n_trimmed_rows, n_gages), real units (m3/s)
    if uselog:
        q_scale = myscaler['q_scale']
        usgsMat_full = np.log(usgs_vals + 1e-4) / q_scale
    else:
        #output_mean/output_std are per-node (length num_nodes); select the gaged columns
        out_mean_g = np.asarray(myscaler['output_mean'])[colIDs]
        out_std_g = np.asarray(myscaler['output_std'])[colIDs]
        usgsMat_full = (usgs_vals - out_mean_g) / out_std_g

    #slice out exactly the rows each loader's local `indx` will reference
    train_usgsMat = torch.FloatTensor(usgsMat_full[noffset:noffset+n_train, :]).to(device)
    val_usgsMat   = torch.FloatTensor(usgsMat_full[trainLen:trainLen+n_val, :]).to(device)
    colIDs_t = torch.LongTensor(colIDs)

    if uselog:
        model_path='/'.join([save_path, f'{model_prefix}_bestmodel_{basestr}_log.pth'])
        model_path_finale='/'.join([save_path,f'{model_prefix}_finalmodel_{basestr}_log.pth'])
    else:
        model_path='/'.join([save_path,f'{model_prefix}_bestmodel_{basestr}.pth'])
        model_path_finale='/'.join([save_path,f'{model_prefix}_finalmodel_{basestr}.pth'])

    if reTrain:
        model.train()
        tune_params = model.parameters()
        optimizer = torch.optim.Adam(tune_params, lr=lr, betas=(0.9, 0.999))
        min_val_loss = np.inf
        l1 = args.train_params.L1Loss

        for epoch in range(nEpoch):
            epochTrainLoss = trainEpoch(model,optimizer,trainLoader,epoch,args,train_usgsMat,colIDs_t,l1,**kwargs)
            epochValLoss   = evalEpoch(model,valLoader,args,val_usgsMat,colIDs_t,l1,**kwargs)
            print("epoch", epoch, ", train loss:", epochTrainLoss, ", val loss:",epochValLoss)
            if epochValLoss < min_val_loss:
                min_val_loss = epochValLoss
                print ('saving  best model...')
                torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), model_path_finale)

    if not usefinal:
        print ('use saved best model ', model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print ('use saved final model ', model_path_finale)
        model.load_state_dict(torch.load(model_path_finale))

    return model

def getLoss(out,usgsSlice,colIDs,indx,l1=True):
    """Masked loss: only counts positions where a real USGS observation exists.
    NaN targets (no observation for that date/gage) are excluded entirely,
    rather than fine-tuning against fabricated/filled values.
    """
    target = usgsSlice[indx,:]
    pred = out[:,colIDs]
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device, requires_grad=True)
    diff = pred[mask] - target[mask]
    if l1:
        return diff.abs().mean()
    else:
        return (diff**2).mean()

def trainEpoch(model,optimizer,loader,epochno,args,usgsMat,colIDs,l1,**kwargs):
    """Do a single fine tune epoch
    """
    model.train()
    clip_norm = True
    n=0
    l_sum=0.0
    starttime = time.time()
    pbar = tqdm.tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epochno}')

    for (x,y),indx in pbar:
        x=x.to(device)
        y=y.to(device)
        model.zero_grad()
        out = model(x).squeeze()
        loss = getLoss(out,usgsMat,colIDs,indx,l1)

        l_sum +=loss.item()
        n+=out.shape[0]
        loss.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.train_params.clipnorm)
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

    print ('time elapsed ', time.time()-starttime)
    return l_sum/n

def evalEpoch(model,loader,args,usgsMat,colIDs,l1,**kwargs):
    model.eval()
    n=0
    l_sum=0.0

    for (x,y),indx in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = model(x).squeeze()

        loss = getLoss(out,usgsMat,colIDs,indx,l1)
        l_sum +=loss.item()
        n+=out.shape[0]
    return l_sum/n

def plotFineTuneResidual(args,nwmMat,finetuneMat,usercmap="rainbow", LOO_gage=None):
    """Visualize node difference due to label propagation
    """
    from nwmutils import plotNetworkResidual
    from genrivernet import loadWatershed

    #hard coding!
    nhdplusdbroot = '/work2/02248/alexsund/maverick2/nwm/nhdplusdb'
    wobj = loadWatershed(args, dbroot=nhdplusdbroot, returnWobj=True)
    comIDset,comIDdict,allDF,goodInd,gageDict = loadWatershed(args, dbroot=nhdplusdbroot, returnWobj=False)
    rootdir = wobj.rootdir
    #remove bad values
    residualVec = np.mean(np.abs(finetuneMat-nwmMat),axis=0)
    residualVec[np.isinf(residualVec)]=0.0

    plotNetworkResidual(
        args.watershed_name, 
        wobj.prettyName, 
        os.path.join(rootdir, wobj.shpfilename), 
        comIDset,
        residualVec, 
        gagelocshp=os.path.join(rootdir, wobj.gagelocfile),
        gageSet=gageDict.keys(), 
        comIDColName=wobj.comIDColName,
        floIDColName=wobj.floIDColName,
        usercmap = usercmap,
        LOO_gage=LOO_gage,
        basinboundshp = os.path.join(rootdir, wobj.shpbasinbound),
        addnwm = args.addnwm,
        uselog = args.data.uselog,
        connect_exta_path = args.connect_exta_path,
        nwm_ver = args.nwm_ver,
        vmin = 0.0, vmax=5.0,
        modeltag=f'{args.modeltag}_finetune',
    )

def checkOtherFlowData(args, gageDict, usgsDict, comIDdict, finetuneMat, nwmMat, fusionMat=None):
    """check data from LBL
    """
    cfs2cms = 0.028316847000000252
    if args.watershed_name=='easttaylor':
        df1 = pd.read_csv('data/easttaylor/385106106571000.txt', delimiter='\t', skiprows=1, header=None,index_col=0)
        df1.columns=['Q']
        #convert to m3/s
        df1['Q'] = df1['Q']*cfs2cms
        df1.index = pd.to_datetime(df1.index,format='%Y-%m-%d',utc=True).date
        df2 = pd.read_csv('data/easttaylor/pump_house.txt', delimiter='\t', skiprows=1, header=None, index_col=0)
        df2.columns=['Q']
        df2.index = pd.to_datetime(df2.index).date
        #df2.index = df2.index + pd.DateOffset(hours=6)
        #flowDict= {1333026:df1,1332754:df2}
        #obsDict = {'385106106571000':1333026, 'pumphouse':1332754}
        obsDict = {'385106106571000':1333026}
        flowDict= {1333026:df1}

    for item in gageDict.keys():
        usgsid = gageDict[item]
        print (f'comid {item}, usgsid {usgsid}, col id {comIDdict[item]}')
        dfQ = usgsDict[usgsid]['Q']    
        dtime = dfQ.index.date
        break

    ntest = finetuneMat.shape[0]
    print ('dtime', dtime[-ntest:])

    for site,comid in obsDict.items():                
        colid = comIDdict[comid]
        df_nwm = pd.DataFrame(nwmMat[:,colid], index=dtime[-ntest:])
        df_nwm.columns=['Qnwm']
        df_mlQ = pd.DataFrame(finetuneMat[:,colid], index= dtime[-ntest:])
        df_mlQ.columns=['Qml']
        if not fusionMat is None:
            df_fusionQ = pd.DataFrame(fusionMat[:,colid], index= dtime[-ntest:])
            df_fusionQ.columns = ['Qfusion']
        
        dfObs = flowDict[comid]
        #do inner join
        if not fusionMat is None:
            df = pd.concat([df_mlQ, df_fusionQ, dfObs, df_nwm],axis=1,join='inner')
        else:
            df = pd.concat([df_mlQ, df_nwm, dfObs],axis=1,join='inner')
        kge_ml = Hydrostats.kge_2009(df['Qml'],df['Q'])
        kge_nwm = Hydrostats.kge_2009(df['Qnwm'],df['Q'])

        if not fusionMat is None:
            kge_fusion = Hydrostats.kge_2009(df['Qfusion'],df['Q'])
            print (site, 'num records', df.shape[0], ' kge_ml_finetune ', kge_ml, ' kge_fusion ', kge_fusion)
        else:
            print (site, 'num records', df.shape[0], ' kge_nwm ', kge_nwm, ' kge_ml_finetune ', kge_ml )
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        ax.plot_date(df['Q'].index, df['Q'], xdate=True, color='gray', label='Obs',linewidth=1.5)
        ax.plot_date(df['Q'].index, df['Qnwm'],'-', xdate=True, color='#DC143C', label='NWM',linewidth=1.5)
        ax.plot_date(df['Q'].index, df['Qfusion'],'-', xdate=True, color='#33A1C9', label='DataFusion',linewidth=1.5)
        ax.set_title(f'Gage: {site}, $KGE_{{NWM}}$={kge_nwm:.3f}, $KGE_{{fusion}}$={kge_fusion:.3f}')
        plt.legend()
        plt.savefig(f'outputs/{args.watershed_name}/othergage_plot{site}.eps')
        plt.close()