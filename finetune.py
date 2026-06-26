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
    #Loop through comID of usgs gages
    colIDs = []
    allDF = [] 
    #note: colIDs should be sorted in ascending order
    for item in gageDict.keys():
        usgsid = gageDict[item]
        print (f'comid {item}, usgsid {usgsid}, col id {comIDdict[item]}')
        dfQ = usgsDict[usgsid]['Q']     
        print (f'Length of usgs record for {usgsid}: {dfQ.shape[0]}')      
        #convert timestamp to date
        #dfQ.index = pd.to_datetime(dfQ.index,format='%Y-%m-%d',utc=True)
        dfQ.index = pd.to_datetime(dfQ.index)
        #dfQ.fillna(inplace=True, method='bfill')  
        colIDs.append(comIDdict[item])
        allDF.append(dfQ)
    allDF = pd.concat(allDF,axis=1, join='outer')
    #allDF may have Nan!!!!!    
    allDF.fillna(inplace=True, method='bfill')  
    print ('allDF shape', allDF.shape)
    #test nan
    print ('nan values', np.where(np.isnan(allDF.values))[0])
    return allDF,colIDs

def disaggregateDaily2Hourly(allUSGSQDF, forcingDF, colIDs, freq=3):
    arr = []
    N = 24/freq
    nrec = forcingDF.shape[0]
    for i in range(0, nrec-N, N):
        df = forcingDF.iloc[i*N:(i+1)*N, colIDs]
        print (df)

def getLoss(out,criterion,usgsMat,colIDs,indx):
    #calculate loss only at observation locations 
    loss = criterion(out[:,colIDs], usgsMat[indx,:])       
    return loss

def finetune(args,model,model_prefix,trainLoader,valLoader,A,save_path,usgsDict, gageDict,comIDdict, basestr, reTrain=False, **kwargs):
    """Main code for fine tuning
    """
    usgsMatO,colIDs = extractGageData(usgsDict,gageDict,comIDdict)

    num_nodes = A.shape[0]    
    nEpoch = args.nepoch
    lr = args.learnrate #this needs to be a small learning rate
    seq = args.seq_length
    usefinal = args.usefinal
    uselog = args.uselog

    #need to transform the data
    if uselog:
        myscaler = pkl.load(open(f'data/{args.watershed_name}/myscaler_log.pkl', 'rb'))  
    else:
        myscaler = pkl.load(open(f'data/{args.watershed_name}/myscaler.pkl', 'rb'))  

    #need to fake other features to use the scaler
    augDimension = num_nodes
    augmat = np.random.randn(usgsMatO.shape[0], augDimension)
    
    #replace at the column locations using observed usgs dta
    augmat[:,colIDs] = usgsMatO

    #Do data normalization using saved scalers from the pretrained model
    if uselog:
        augmat = myscaler['pt'].transform(augmat)
    else:        
        augmat = (augmat-myscaler['output_mean'])/myscaler['output_std']

    #!!!! offset the data
    usgsMat = augmat[seq:, colIDs]

    colIDs = torch.LongTensor(colIDs)
    usgsMat = torch.FloatTensor(usgsMat).to(device)

    if uselog:
        model_path='/'.join([save_path, f'{model_prefix}_bestmodel_{basestr}_log.pth'])
        model_path_finale='/'.join([save_path,f'{model_prefix}_finalmodel_{basestr}_log.pth'])
    else:
        model_path='/'.join([save_path,f'{model_prefix}_bestmodel_{basestr}.pth'])
        model_path_finale='/'.join([save_path,f'{model_prefix}_finalmodel_{basestr}.pth'])

    if reTrain:
        model.train()

        """
        tune_params = [
                 {'params': model.end_conv_2.parameters(), 'lr': lr},
            ]
        """
        tune_params =  model.parameters()
        optimizer = torch.optim.Adam(
                tune_params, 
                lr=lr, 
                betas=(0.9, 0.999)
            )

        min_val_loss = np.inf

        if args.L1Loss:
            lossfun = torch.nn.L1Loss()
        else:
            lossfun = torch.nn.MSELoss()


        for epoch in range(nEpoch):
            epochTrainLoss = trainEpoch(model,optimizer,trainLoader,lossfun,epoch,args,usgsMat,colIDs,**kwargs)
            epochValLoss   = evalEpoch(model,valLoader,lossfun,args,usgsMat,colIDs,**kwargs)
            print("epoch", epoch, ", train loss:", epochTrainLoss, ", val loss:",epochValLoss)
            if epochValLoss < min_val_loss:
                #only start to record best models after 10 epochs
                min_val_loss = epochValLoss
                print ('saving  best model...')
                torch.save(model.state_dict(), model_path)
        #save the final model                
        torch.save(model.state_dict(), model_path_finale)            

    if not usefinal:
        print ('use saved best model ', model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print ('use saved final model ', model_path_finale)
        model.load_state_dict(torch.load(model_path_finale))

    return model

def trainEpoch(model,optimizer,loader,criterion,epochno,args,usgsMat,colIDs,**kwargs):
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
        y=y.to(device)     # (batch, nnode)    
        model.zero_grad()  # Clear gradients.
        out = model(x).squeeze() #(batch, nnode)
        loss = getLoss(out,criterion,usgsMat,colIDs,indx)

        l_sum +=loss.item()
        n+=out.shape[0]
        loss.backward()  # Derive gradients.
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipnorm)
        optimizer.step()  # Update parameters based on gradients.
        pbar.set_postfix(loss=loss.item())

    print ('time elapsed ', time.time()-starttime)
    return l_sum/n    

def evalEpoch(model,loader,criterion,args,usgsMat,colIDs, **kwargs):
    model.eval()
    n=0
    l_sum=0.0
    
    for (x,y),indx in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = model(x).squeeze()  # Perform a single forward pass.

        loss = getLoss(out,criterion,usgsMat,colIDs,indx)
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
        uselog = args.uselog,
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