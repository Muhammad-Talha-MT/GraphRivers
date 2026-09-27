#author: alex sun
#date: 02152021
#date: 03012021, finalize for production runs
#date: 03032021, replace the optimizer
#date: 03292021, use the original graphwavenet by wu
#date: 05312021, adapt for physics-based GNN
#!!!!conda env!!!!!!!!!!!
#>conda activate waterml
#date: 09212021, change seq length
#date: 10182021, revisiting the code. Change seq len to 120 days and 180 days
#date: 1/08/2022, clean up the code
#date: 2/18/2022, revised uselog
#date: 2/19/2022, final clean up [original version]
#date: 4/18/2022, modify for musikal/houston model
#date: 05/01/2022, change to hourly using nldas data
#date: 09/14/2022, add aorc precip/air temp data and revisit
#date: 04/13/2023, cleanup for sharing
#Three steps
#Step 1. >. rungenriver #this step needs to be run on ls6 [this runs genrivernet.py]
#Step 2. >. runloadrivernet
#Step 3. >. rungwnet [this is equivalent to python trainwavenetwu_hourly.py --config config_houston.yaml ] 
#Step 4 Postprocessing
#=============================================================================
#third-party imports
import random
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import argparse
import os,sys
import time
import hydrostats as Hydrostats
import tqdm

#custom import from this repo
from loadrivernet import loadNWMDF,formAdjacencyMat,genMLDataSetsWithForcing
from genrivernet import loadWatershed
from nwmutils import plotNetworkNSE,getUSGSData,compareNWMUSGS
from basinheatmap import pltNetworkgraph
from finetune import finetune, plotFineTuneResidual
from gwnetmodel import GWNet
from util_gtnet import Optim, load_adj, load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getDataLoaders(watershed,in_dim, batchsize,seq,comIDSet,forcingDF,allDF,outputdir,
                   uselog=False,addstatics=False,addnwm=False,removeSWE=False):
    """wrapper for loading data loaders
    """
    trainDataset,valDataset,testDataset,nfeatures = \
        genMLDataSetsWithForcing(watershed,in_dim,comIDSet,forcingDF,allDF,seq,outputdir,trainRatio=(0.7,0.15),
        logTran=uselog,addStatics=addstatics,addNWM=addnwm,removeSWE=removeSWE, 
        useCustomDataset=True,
        trainEndDate='2010-12-31',
        valEndDate='2015-12-31')

    trainLoader = DataLoader(trainDataset, batch_size=batchsize, shuffle=True, drop_last=True,num_workers=4)
    valLoader = DataLoader(valDataset, batch_size=batchsize, shuffle=False, drop_last=True,num_workers=4)
    testLoader = DataLoader(testDataset, batch_size=batchsize, shuffle=False, drop_last=False)

    return trainLoader,valLoader,testLoader,nfeatures 

def getLoss(args, out, y, criterion,**kwargs):
    """Loss function
    """
    if args.data.uselabel:
        #do fine-tune using observation data 
        gagecols = torch.LongTensor(kwargs['gagecols'])
        unlabeled = torch.LongTensor(kwargs['unlabledcols'])
        #assume the last dimension is nnode!!!
        loss1 = criterion(out[:,gagecols], y.detach()[:,gagecols])
        loss2 = criterion(out[:,unlabeled],y.detach()[:,unlabeled])
        #print (loss1.item(), loss2.item())
        loss = args.fine_tune.lambdawt*loss1+loss2
    else:
        #do pre-training
        loss = criterion(out, y.detach())    
    return loss

def trainEpoch(model,optimizer,loader,criterion,epochno,args,**kwargs):
    """Train an epoch
    Params
    ------
    model, model to be trained
    optimizer, solver to use
    loader, dataloader object
    criterion, loss funcdtion
    epochno, the current epoch number
    args, **kwargs, optional settings
    """    
    model.train()
    clip_norm = True
    n=0
    l_sum=0.0
    starttime = time.time()
    pbar = tqdm.tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epochno}')

    import os as _os
    _debug_max_batches = int(_os.environ.get('DEBUG_MAX_BATCHES', 0))
    _batch_count = 0

    for (x,y),indx in pbar: 
        _batch_count += 1
        if _debug_max_batches > 0 and _batch_count > _debug_max_batches:
            print(f'DEBUG: stopping after {_debug_max_batches} batches')
            break
        x=x.to(device)     
        y=y.to(device)           #(batch, nnode)    

        model.zero_grad()        
        out = model(x).squeeze() #(batch, nnode)
        
        loss = getLoss(args,out,y,criterion,**kwargs)

        l_sum +=loss.item()
        n+=out.shape[0]
        loss.backward()  
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.train_params.clipnorm)
        optimizer.step()  # Update parameters based on gradients.

    print ('time elapsed ', time.time()-starttime)
    return l_sum/n    

def evalEpoch(model,loader,criterion,args,**kwargs):
    """Evaluate epoch
    """
    model.eval()
    n=0
    l_sum=0.0
    
    for (x,y),indx in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = model(x).squeeze()  # Perform a single forward pass.

        loss = getLoss(args,out,y,criterion,**kwargs)
        l_sum +=loss.item()
        n+=out.shape[0]
    return l_sum/n    

def train(args, trainLoader, valLoader, A, in_dim, save_path, 
        out_dim=1, reTrain=False, **kwargs):
    """
    Parameters:
    ---------
    num_nodes: number of graph nodes
    in_dim: feature dim of input
    out_dim: number of prediction steps  (prediction length, t+1, t+2,...)
    addaptadj: whether to add apt adj matrix (eq 6) to graph conv layer
    apt_size: size of latent dim for randomly initializing node embedding

    """    
    num_nodes = A.shape[0]
    
    nEpoch = args.train_params.nepoch
    lr = args.train_params.learnrate
    seq = args.data.seq_length
    uselog = args.data.uselog

    addaptadj = args.gwnet_params.addaptadj

    print ('in_dim', in_dim, 'out_dim', out_dim)
    adj_mx = load_adj(A, args.gwnet_params.adjtype)

    if args.gwnet_params.aptonly:
        supports = None 
    else:
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        print ('support len', len(supports))
    
    #for 30, use  blocks=7,layers=2, kernel_size =4
    #for 120, use blocks=4,layers=4, kernel_size=4
    #for 365, use kernel_size=8
    if seq == 30:
        blocks = 7
        layers = 2
        kernel_size = 4
    elif seq == 60:
        blocks = 4
        layers = 4
        kernel_size = 4
    elif seq == 90: 
        blocks = 4
        layers = 4
        kernel_size = 4
    elif seq == 120: 
        blocks = 4
        layers = 4
        kernel_size = 4
    elif seq == 180:
        blocks = 3
        layers = 4        
        kernel_size = 6
    elif seq == 365:
        blocks = 4
        layers = 4        
        kernel_size = 8
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
        skip_channels=args.gwnet_params.nhid * 4, #original *8 
        end_channels=args.gwnet_params.nhid * 8, #original *16
        apt_size = args.gwnet_params.apt_size,
        blocks = blocks,
        layers = layers,
        kernel_size= kernel_size,seq_len=seq) #must change kernel_size to 4 to make this work

    seed = args.seed
    #the following lines set up output names 
    if args.train_params.L1Loss:
        lossfunstr = 'L1'
    else:
        lossfunstr = "L2"
    if addaptadj:
        basestr = f'seq{seq}_{lossfunstr}_seed{seed}_node{num_nodes}_aptadj_nwm{args.data.nwm_ver}_{args.data.forcing_source}_{args.data.interval}'
    else:
        basestr = f'seq{seq}_{lossfunstr}_seed{seed}_node{num_nodes}_nwm{args.data.nwm_ver}_{args.data.forcing_source}_{args.data.interval}'
    if args.data.addnwm:
        basestr = basestr+'_usenwm'
    if args.data.uselabel:
        basestr = basestr+'_label'
    if args.network.directed_adj:
        basestr = basestr+'_directed'
    if args.network.weighted_adj:
        basestr = basestr+'_weighted'
    if args.network.connect_exta_path:
        basestr = basestr+'_extrapath'

    if uselog:
        model_path='/'.join([save_path, f'gwnet2bestmodel_{basestr}_log.pth'])
    else:
        model_path='/'.join([save_path,f'gwnet2bestmodel_{basestr}.pth'])

    model.to(device)    
    if reTrain:
        model.train()
        optimizer = Optim(
                model.parameters(), 'adamw', lr, clip=args.train_params.clipnorm, lr_decay=1e-5, start_decay_at=15)

        min_val_loss = np.inf
        if args.train_params.L1Loss:
            lossfun = torch.nn.L1Loss()
        else:
            lossfun = torch.nn.MSELoss()

        for epoch in range(nEpoch):
            epochTrainLoss = trainEpoch(model,optimizer,trainLoader,lossfun,epoch,args,**kwargs)
            epochValLoss = evalEpoch(model,valLoader,lossfun,args,**kwargs)
            print("epoch", epoch, ", train loss:", epochTrainLoss, ", val loss:",epochValLoss)
            if epochValLoss < min_val_loss and epoch>10:
                #only start to record best models after 10 epochs
                min_val_loss = epochValLoss
                print ('saving  best model')
                torch.save(model.state_dict(), model_path)

    print ('use saved best model ', model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def test(args,model,testLoader,nNode,invTransform=True,reGen=True):
    """Test routine
    Params
    ------
    reGen, set True to generate test data at all nodes
    """
    datafile = 'data/{0}/testmats{1}.pkl'.format(args.watershed_name, nNode)

    if reGen:
        
        uselog = args.data.uselog
        watershed_name = args.watershed_name
        
        model.eval()

        if uselog:
            myscaler = pkl.load(open(f'data/{watershed_name}/myscaler_log.pkl', 'rb'))  
        else:
            myscaler = pkl.load(open(f'data/{watershed_name}/myscaler.pkl', 'rb'))  

        outMat = []
        trueMat = []
        
        for (x,y),indx in testLoader:
            x = x.to(device)
            
            with torch.no_grad():
                out = model(x).squeeze()            
                out = out.data.cpu().numpy()

            y = y.data.cpu().numpy()
            if invTransform:
                if args.data.uselog:
                    #inverse transform
                    q_scaler = myscaler['q_scale']
                    y = np.exp(y*q_scaler)-1e-4
                    out = np.exp(out*q_scaler)-1e-4
                else:
                    #map to the original space
                    y = y*myscaler['output_std']+myscaler['output_mean']
                    out = out*myscaler['output_std']+myscaler['output_mean']
                #reset invalid values
                out[out<0]=0.0
                y[y<0.0] = 0.0
                #0303
                if len(np.where(np.isnan(out))[0])>0:
                    print ('removed ', len(np.where(np.isnan(out))[0]), ' bad values')  
                out[np.isnan(out)]=0.            

            outMat.append(out)
            trueMat.append(y)
        outMat  = np.concatenate(outMat)    
        trueMat = np.concatenate(trueMat)
        pkl.dump([trueMat,outMat],open(datafile, 'wb'))
    else:
        trueMat,outMat = pkl.load(open(datafile, 'rb'))
    #       nwm      ml
    return trueMat,outMat

def getNSE(trueMat,simMat):
    """Calculate NSE metric
    """
    print ('test mat shape', trueMat.shape)
    if len(trueMat.shape)>1:
        nNode = trueMat.shape[1]
    else:
        nNode = 1
        trueMat = trueMat[:, np.newaxis]
    nse=np.zeros((nNode,))

    for i in range(nNode):
        df = pd.DataFrame(np.c_[simMat[:,i],trueMat[:,i]], columns=('qsim','qobs'))
        df = df.dropna()            
        #special treatment for low-flow reaches
        if (df['qobs'].max()==df['qobs'].min()):
            df['qobs'] = df['qobs'].to_numpy()+1e-3*np.random.randn(df.shape[0])
            
        nse[i]= Hydrostats.nse(df['qsim'],df['qobs'])
        
    print (f'median nse {np.median(nse):.3f}, mean nse {np.mean(nse):.3f}, max nse {np.max(nse):.3f}, min nse {np.min(nse):.3f}')
    return nse

def doCompareNWM_ML_USGS(args,comIDdict,gageDict,usgsDict,nwmMat,mlMat,allDF,modelName,interval='24H',usgsvar='dv'):
    """Compare NWM, ML, and USGS 
    Params
    ------
    comIDdict, map from comID to node index
    gageDict, map from comID to usgs gageid
    usgsDict, dictionary of streamflow DF, key=usgs gageid
    10/9/2022: change usgs data to 3H
    """
    ind = allDF.index
    if interval in ['1D', '1d', '24H']:
        convert2daily = True
    else:
        convert2daily = False

    for comID in gageDict.keys():
        gageID = gageDict[comID]
        
        fig,ax = plt.subplots(1,1,figsize=(16,16))
        dfUSGS = usgsDict[gageDict[comID]]['Q']           
        if convert2daily and usgsvar == 'iv':
            #convert timestamp to date
            dfUSGS = dfUSGS.resample('1d').mean()            
            dfUSGS.index = pd.to_datetime(dfUSGS.index,format='%Y-%m-%d',utc=True)    
            dfUSGS.fillna(inplace=True, method='bfill')  
        
        #uncomment the following to debug
        #np.savetxt('data/{1}/{0}.txt'.format(comID, args.watershed_name),residual)

        #get nwm by column id
        nwmvec = nwmMat[:,comIDdict[comID]]
        #get ml by column id
        mlvec = mlMat[:,comIDdict[comID]]
        ntest = len(mlvec); print(ntest)
        df_ML  = pd.DataFrame(mlvec, index=ind[-ntest:])
        df_NWM = pd.DataFrame(nwmvec, index=ind[-ntest:])



        if convert2daily:
            df_ML = df_ML.resample('1d').mean()
            df_NWM = df_NWM.resample('1d').mean()
        
        dfUSGS = dfUSGS.iloc[-df_ML.shape[0]:]

        data = {'USGS': dfUSGS.values.squeeze(), 'NWM':df_NWM.values.squeeze(),'ML':df_ML.values.squeeze()}
        dfQ = pd.DataFrame(data, index=dfUSGS.index)
        
        ax.plot(dfQ.index, dfQ['USGS'], '-', color='gray', alpha=0.7,label='Obs')
        ax.plot(dfQ.index, dfQ['NWM'],'-', color='coral', label='NWM')
        ax.plot(dfQ.index, dfQ['ML'],'-', color='blue', label='ML')
        #ax.set_yscale("log")  
        ax.set_xlabel('Date')
        ax.set_ylabel('Q (m3/s)')
        plt.grid(True)
        
        #qsim, qobs
        #calculate NSE and Pearson's correlation
        nse_nwm = Hydrostats.nse(dfQ.iloc[:,1],dfQ.iloc[:,0])
        corr_nwm = Hydrostats.pearson_r(dfQ.iloc[:,1],dfQ.iloc[:,0])
        nse_ml = Hydrostats.nse(dfQ.iloc[:,2],dfQ.iloc[:,0])
        corr_ml = Hydrostats.pearson_r(dfQ.iloc[:,2],dfQ.iloc[:,0])

        plt.title(f'Gage {gageID}\nNWM_NSE={nse_nwm:.3f},NWM_R={corr_nwm:.3f}\nML_NSE={nse_ml:.3f},ML_R={corr_ml:.3f}', fontsize=14)
        plt.legend()
        pngname = 'outputs/{0}/plt_{3}_{0}_{1}_seq{2}_nwm{4}_{5}_{6}'.format(args.watershed_name,gageID,args.data.seq_length,
                     modelName, args.data.nwm_ver, args.data.forcing_source, args.data.interval)
        if args.network.directed_adj:
            pngname = pngname +'_directed'
        if args.network.weighted_adj:
            pngname = pngname +'_weighted'
        if args.data.uselabel:
            pngname = f'{pngname}_holdout{args.data.holdoutcomid}'

        plt.savefig('{0}.png'.format(pngname))
        plt.close()


def compareNSE(args, nse1, nse2):
    from statsmodels.distributions.empirical_distribution import ECDF

    plt.figure()
    ecdf1 = ECDF(nse1, side='right')
    ecdf2 = ECDF(nse2, side='right')
    plt.plot(ecdf1.x,ecdf1.y, 'b-', label='no update')
    plt.plot(ecdf2.x,ecdf2.y, 'r-', label='w/ update')
    plt.legend(loc='best')
    plt.savefig('data/{0}/{0}_compare_nse.png'.format(args.watershed_name))
    plt.close()

def plotNSE(args,nsevec,vmin=-1, vmax=1, usercmap="rainbow"):    
    wobj = loadWatershed(args, returnWobj=True)
    comIDset,comIDdict,allDF,goodInd,gageDict = loadWatershed(args, returnWobj=False)
    rootdir = wobj.rootdir
    plotNetworkNSE(
        args.watershed_name, 
        wobj.prettyName, 
        os.path.join(rootdir, wobj.shpfilename), 
        comIDset,nsevec, 
        gagelocshp=os.path.join(rootdir, wobj.gagelocfile),
        gageSet=gageDict.keys(), 
        comIDColName=wobj.comIDColName,
        floIDColName=wobj.floIDColName,
        vmin=vmin, vmax=vmax, usercmap = usercmap,
        exportShpFile=True,
        nwm_ver= args.data.nwm_ver,
        forcing_source=args.data.forcing_source,
        interval=args.data.interval,
    )

def replaceLabelInDF(allDF, usgsDict, gageDict,comIDdict):
    """Replace NWM output at gageloc using observed streamflow data
    """
    #Loop through comID of usgs gages
    allDF.index = pd.to_datetime(allDF.index,format='%Y-%m-%d',utc=True)
    for item in gageDict.keys():
        usgsid = gageDict[item]
        print (f'comid {item}, usgsid {usgsid}, col id {comIDdict[item]}')
        #get column index in allDF
        nwm = allDF[item]
        dfQ = usgsDict[usgsid]['Q']     
        print (f'len of usgs record for {usgsid}: {dfQ.shape[0]}')      
        #convert timestamp to date
        dfQ.index = pd.to_datetime(dfQ.index,format='%Y-%m-%d',utc=True)
        allDF[item] = dfQ

    print ('# of nan values before filling', allDF.isna().sum().sum())
    allDF.fillna(inplace=True, method='bfill')        
    print ('# of nan values after filling', allDF.isna().sum().sum())
    return allDF

def CV(args, holdoutComID, adjmat,nwmMat, usgsDict, gageDictFull,comIDdict):
    """perform cross-validation on the holdout gage and on its neighbors
    Params
    ------
    holdoutComID, comid of usgs gage held out 
    adjmat, adj matrix
    allDF, NWM data
    usgsDict, map of usgs gage to usgs obs dataframe
    gageDict, map from comid to usgs gageid
    """
    #load ML results
    if args.network.weighted_adj:
        mlMatlabeled = pkl.load(open('outputs/{0}/mlmat_labeled_weighted.pkl'.format(args.watershed_name), 'rb'))
        mlMatunLabeled = pkl.load(open('outputs/{0}/mlmat_weighted.pkl'.format(args.watershed_name), 'rb'))
    else:
        mlMatlabeled = pkl.load(open('outputs/{0}/mlmat_labeled.pkl'.format(args.watershed_name), 'rb'))
        mlMatunLabeled = pkl.load(open('outputs/{0}/mlmat.pkl'.format(args.watershed_name), 'rb'))

    #compare obs, NWM, ML at the holdout location
    holdoutCol = comIDdict[holdoutComID]
    print ('hold out gage', gageDictFull[holdoutComID])
    dfQ = usgsDict[gageDictFull[holdoutComID]]['Q']           
    #convert timestamp to date
    dfQ.index = pd.to_datetime(dfQ.index,format='%Y-%m-%d',utc=True)    
    dfQ.fillna(inplace=True, method='bfill')  
    obsvec = dfQ.to_numpy()
    print ('len of obs data', len(obsvec))

    nwmvec = nwmMat[:,holdoutCol]
    mlvecUnlabeled = mlMatunLabeled[:,holdoutCol] 
    mlvecLabeled = mlMatlabeled[:,holdoutCol]
    
    #get the data for testing period
    obsvec=obsvec[-len(nwmvec):]
    print ('nwm vs. usgs', Hydrostats.nse(nwmvec,obsvec))    
    print ('unlabeled ml vs. usgs', Hydrostats.nse(mlvecUnlabeled,obsvec))
    print ('labeled ml vs. usgs', Hydrostats.nse(mlvecLabeled,obsvec))

    for item in gageDictFull.keys():        
        nwmvec = nwmMat[:,comIDdict[item]]
        dfQ = usgsDict[gageDictFull[item]]['Q']           
        #convert timestamp to date
        dfQ.index = pd.to_datetime(dfQ.index,format='%Y-%m-%d',utc=True)    
        dfQ.fillna(inplace=True, method='bfill')  
        obsvec = dfQ.to_numpy()
        print ('len of obs data', len(obsvec))
        obsvec = obsvec[-len(nwmvec):]        
        mlvecUnlabeled = mlMatunLabeled[:,comIDdict[item]] 
        print (f"----GAGE {item}----")
        print ('nwm vs. usgs', Hydrostats.nse(nwmvec,obsvec))    
        print ('unlabeled ml vs. usgs', Hydrostats.nse(mlvecUnlabeled,obsvec))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse arguments for genrivernet",
    )
    parser.add_argument('--config', type=str, required=True, help='config file name')
    cargs = parser.parse_args()
    args = load_config(cargs.config)
    
    #09072023, to be removed this has been moved to config file
    #parser = get_shared_arg_parser() 
    #add additional arguments
    # parser.add_argument('--watershed_name', type=str, default="houston",help='specify watershed name')
    # parser.add_argument('--compare_to_usgs', action='store_true', default=False, help='compare to usgs record')
    # parser.add_argument('--compare_to_prms', action='store_true', default=False, help='compare to prms results')

    # parser.add_argument("--retrain",action='store_true', default=False, help="retrain the model only if true" )
    # parser.add_argument("--nepoch", type=int, default=60, help="set the number of epochs")
    # parser.add_argument("--learnrate",type=float,default=0.01, help="learning rate")
    # parser.add_argument("--seed", type=int, default=20210221, help="random seed") 
    # parser.add_argument("--uselog", action="store_true", default=False, help="true to use log transform")
    # parser.add_argument("--L1Loss", action="store_true", default=False, help="true to use L1Loss function")
    # parser.add_argument("--usefinal", action="store_true", default=False, help="true to use the final saved model")
    
    # parser.add_argument("--uselabel", action="store_true", default=False, help="true to use usgs obs at measurement locs")
    # parser.add_argument("--regen_dataloader", action="store_true", default=False, help="reload data loaders")
    # parser.add_argument("--lambdawt", type=float, default=1.0, help="weight for loss function")
    # parser.add_argument("--holdoutcomid",type=int, default=0, help="comid for cv test")
    # parser.add_argument("--weighted_adj",action="store_true", default=False, help="true to generate weight adj mat")
    # parser.add_argument("--directed_adj",action="store_true", default=False, help="true to generate weight adj mat")
    # parser.add_argument("--addstatics",action="store_true", default=False, help="true to generate weight adj mat")
    # parser.add_argument("--addnwm",action="store_true", default=False, help="add antecedent nwm states")
    # parser.add_argument("--regen_testdata",action="store_true", default=False, help="regen test data")
    # parser.add_argument("--connect_exta_path",action="store_true", default=False, help="true to connect extra paths")
    # parser.add_argument("--cutoffstattype",type=int, default=1, help="1 median, 2 min for flow cutoff")
    # parser.add_argument('--nwm_ver', type=str, required=True, default='2.0', help='specify watershed name')

    # parser.add_argument("--tune_omega",action="store_true", default=False, help="true to tune omega")
    # parser.add_argument("--omega", type=float, default=500, help="omega for label propagation")    
    # parser.add_argument("--omega_lo", type=float, default=50, help="lower bound for omega tune")    
    # parser.add_argument("--omega_hi", type=float, default=2000, help="upper bound for omega tune")    
    # parser.add_argument("--omega_stepsize", type=float, default=50, help="step size for omega tune") 
    # #   
    # parser.add_argument("--normalize_lp",action="store_true", default=False, help="true to normalize residual for label propagation")
    # parser.add_argument("--lp_task", type=int, default=1, help="task=1 LOO, task=2, use all ") 
    # parser.add_argument("--modeltag", type=str, default="gwnet")
    # parser.add_argument("--interval", type=str, default="24H", help="interval for aggregating nwm data (need to be compatible with forcing)")
    # parser.add_argument("--forcing_source", type=str, default="daymet", help="forcing data source)")
    # #
    # parser.add_argument("--retune",action='store_true', default=False, help="redo finetune only if true" )    
    # args = parser.parse_args()
    # print ('options ', args)

    #
    #set random seed    
    #
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    save_path = os.path.join(args.rootdir, 'models/{0}'.format(args.watershed_name))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    seq = args.data.seq_length
    t0 = time.time()
    #
    #load NWM dataframe (generated by using loadrivernet.py, so regen=False)
    #
    comIDset,comIDdict,allDF,goodInd,gageDict = loadNWMDF(args)    
    #
    #load USGS data
    #the data in usgsDict has the same order as gageDict
    #[switch reLoad to True to reload usgs gage streamflow]
    if args.data.interval == '3H':
        usgsDict = getUSGSData(args.watershed_name,gageDict.values(),
                           reLoad=args.data.reload_usgs, valueType='iv')
    else:
        usgsDict = getUSGSData(args.watershed_name,gageDict.values(),
                           reLoad=args.data.reload_usgs, valueType='dv')

    #
    #replace nwm data at usgs gageloc if required
    #
    kwargs={}
    originalgageDict = gageDict.copy()
   
    #uncomment to plot NWM vs. USGS
    #doCompareNWMUSGS(args,comIDdict,gageDict,usgsDict)

    print ('loaded dataframe ', allDF.shape)
    print ('elapsed time ', time.time()-t0)
    t0 = time.time()
    #
    #load adj mat
    #
    A = formAdjacencyMat(args.rootdir,
                args.watershed_name,
                goodInd,comIDdict,
                isWeighting=args.network.weighted_adj,
                isDirected=args.network.directed_adj,
                connect_exta_path=args.network.connect_exta_path,
                add_self_loop=True, 
                nwm_ver=args.data.nwm_ver )

    nnode = A.shape[0]
    
    #UNcomment this to generate the heatmap 
    #pltNetworkgraph(args, A, comIDdict,  graphtype='full', comIDColName='COMID',
    #    flowlineshp='houston_flowline_vaa.shp',h12shp='houston_huc12.shp',basinboundshp='houstonproj.shp')        
    
    outputdir = os.path.join(args.rootdir, 'data/{0}'.format(args.watershed_name))
    #load forcing
    if args.data.forcing_source in ['daymet', 'Daymet']:
        forcingDF = pkl.load(open('data/{0}/daymetforcing{1}_nwm{2}.pkl'.format(args.watershed_name,nnode, args.data.nwm_ver),'rb'))    
        removeSWE = args.data.removeSWE
    elif args.data.forcing_source in ['nldas', 'NLDAS']:
        forcingDF = pkl.load(open('data/{0}/nldasforcing{1}_nwm{2}_{3}.pkl'.format(args.watershed_name,nnode, args.data.nwm_ver, args.data.interval),'rb'))    
        removeSWE = False
    elif args.data.forcing_source in ['aorc', 'AORC']:
        forcingDF = pkl.load(open('data/{0}/aorcforcing{1}_nwm{2}_{3}.pkl'.format(args.watershed_name,nnode, args.data.nwm_ver, args.data.interval),'rb'))    
        removeSWE = False
    elif args.data.forcing_source in ['era5', 'ERA5']:
        forcingDF = pkl.load(open('data/{0}/era5forcing_full_nwm{1}_{2}.pkl'.format(args.watershed_name, args.data.nwm_ver, args.data.interval),'rb'))
        removeSWE = False
    else:
        raise ValueError("invalid forcing source")
    trainLoader,valLoader,testLoader,nfeatures = getDataLoaders(
        watershed=args.watershed_name,
        in_dim = args.gwnet_params.in_dim,
        batchsize=args.train_params.batch_size, 
        seq=seq, 
        comIDSet = comIDset,
        forcingDF=forcingDF, 
        allDF=allDF,
        outputdir=outputdir,
        uselog=args.data.uselog,
        addstatics=args.data.addstatics,
        addnwm = args.data.addnwm,
        removeSWE= removeSWE
    )

    print ('loaded data from ', outputdir)
    print ('elapsed time ', time.time()-t0)
     
    t0 = time.time()
    #
    #train model
    #
    model = train(args, trainLoader,valLoader,A, nfeatures, save_path=save_path, 
                reTrain=args.train_params.retrain,**kwargs)
    #
    #test model
    #
    print ('Get testing results ...')
    trueMat,outMat = test(args, model,testLoader,nnode,invTransform=True,reGen=True) #nwm, ML results
        
    dataout = f'gwnet_seq{args.data.seq_length}_seed{args.seed}_node{nnode}_nwm{args.data.nwm_ver}'
    if args.data.addnwm:
        dataout = dataout+ "_usenwm"
    if args.data.uselog:
        dataout = dataout + "_uselog"

    #saving ensemble outputs
    os.makedirs(f'ensemble_out/{args.watershed_name}',exist_ok=True)
    dataout=os.path.join(f'ensemble_out/{args.watershed_name}', dataout)
    pkl.dump((trueMat,outMat), open('{0}.pkl'.format(dataout), 'wb'))

    #turn genNSE on to output shp file for web app 
    genNSE= True
    if genNSE:
        nsevecF = getNSE(trueMat,outMat)    
        plotNSE(args, nsevecF)
        print ('elapsed time ', time.time()-t0)

    t0 = time.time()
    
    doCompareNWM_ML_USGS(args, comIDdict, gageDict, usgsDict, nwmMat=trueMat, mlMat=outMat, 
                    allDF=allDF, modelName="gwmodel", interval= args.data.interval, usgsvar='iv')

    if args.train_params.retune:
        #
        #fine tuning
        #
        save_path = os.path.join(args.rootdir, 'models/{0}'.format(args.watershed_name))
        if args.train_params.L1Loss:
            lossfunstr = 'L1'
        else:
            lossfunstr = "L2"
        
        num_nodes = A.shape[0]
        if args.gwnet_params.addaptadj:
            basestr = f'seq{seq}_{lossfunstr}_seed{args.seed}_node{num_nodes}_aptadj_nwm{args.data.nwm_ver}'
        else:
            basestr = f'seq{seq}_{lossfunstr}_seed{args.seed}_node{num_nodes}_nwm{args.data.nwm_ver}'

        if args.data.addnwm:
            basestr = basestr+'_usenwm'
        if args.data.uselabel:
            basestr = basestr+'_label'
        if args.network.directed_adj:
            basestr = basestr+'_directed'

        model = finetune(args,model,"gwnetfinetune",trainLoader,valLoader,A,
                save_path,usgsDict, gageDict,comIDdict, basestr, 
                reTrain=args.train_params.retune, **kwargs)

        #
        #get test on finetuned ML
        #
        print ('Get testing results ...')
        
        trueMat,finetuneMat = test(args, model, testLoader,nnode,invTransform=True, reGen=True) #nwm, ML results
        print (finetuneMat.shape)
        plotFineTuneResidual(args,trueMat,finetuneMat,usercmap="rainbow")
        doCompareNWM_ML_USGS(args,comIDdict,gageDict,usgsDict,trueMat,finetuneMat,allDF=allDF, modelName="gwfinetune")

if __name__ == '__main__':
    main()
