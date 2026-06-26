#author: Alex Sun
#purpose: driver for label propagation
#date: 03032022, updated for final paper results
#
import pickle as pkl
import numpy as np
import pandas as pd
import hydrostats as Hydrostats
from lp_util import lp_refine,OrderedSet
import torch
import os,sys
from genrivernet import loadWatershed
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

def doLOO(args,device,adjmat,nwmMat,mlMat,r_L,obsData, gageNodeInd, gageCOMId,gageDict,
            isPlotResidual=False, **kwargs):
    
    #do leave-one-out CV
    if args.tune_omega:
        #Tune omega    
        OMEGA = np.arange(args.omega_lo, args.omega_hi, args.omega_stepsize)      
    else:
        OMEGA = [args.omega]  
    
    bestOmega=OMEGA[0]
    bestCV_NSE=-100.0
    prev_nse = 0.0
    #save a copy of adjmat
    adjmat0 = adjmat.copy()

    for omega in OMEGA:
        testdict={}    #dictionary to store results

        for indx,anode in enumerate(gageNodeInd):            
            if np.sum(adjmat0[anode,:])==0:
                print ('BAD OBS NODE')
            uMat = mlMat.copy() #uMat only has test data (i.e., nTest implied)
            cv_gageid = [indx] #this is the colum index in sorted gageid

            traingageid = list(range(len(gageNodeInd))) #relative to the obs set
            testgageid = []
            
            for item in cv_gageid:
                traingageid.remove(item)                
                testgageid.append(gageNodeInd[item]) #relative to the whole node set
            

            for id1,id2 in zip(testgageid,cv_gageid):

                kgeval  = Hydrostats.kge_2009(uMat[:,id1],obsData[:,id2])
                kgeval2 = Hydrostats.kge_2009(nwmMat[:,id1],obsData[:,id2])
                maeval  = Hydrostats.mae(uMat[:,id1],obsData[:,id2])
                maeval2 = Hydrostats.mae(nwmMat[:,id1],obsData[:,id2])
                nseval  = Hydrostats.nse(uMat[:,id1],obsData[:,id2])
                nseval2 = Hydrostats.nse(nwmMat[:,id1],obsData[:,id2])
                corrval = Hydrostats.pearson_r(uMat[:,id1],obsData[:,id2])
                corrval2= Hydrostats.pearson_r(nwmMat[:,id1],obsData[:,id2])
                testdict[id1] = {
                    'model_before':uMat[:,id1].copy(),
                    'usgs' : obsData[:,id2],
                    'nse_before' : nseval,
                    'nse_nwm'    : nseval2,
                    'mae_before' : maeval,
                    'mae_nwm'    : maeval2,
                    'kge_before' : kgeval,
                    'kge_nwm'    : kgeval2,
                    'corr_before': corrval,
                    'corr_nwm'   : corrval2,
                }

            #remove testing node from gagenode
            gageNodeIndTrain = np.array(list(OrderedSet(gageNodeInd) - OrderedSet(testgageid)))

            #only use train as labeled (remove holdout)
            r_L_train = r_L[:,traingageid]            

            idx = np.arange(adjmat0.shape[0])
            idx_unlabeled = np.array(list(OrderedSet(idx) - OrderedSet(gageNodeIndTrain)))
            idx_obs = np.array(gageNodeIndTrain, dtype=np.int16)

            orphan_nodes=[]
            for i in range(adjmat0.shape[0]):
                if np.sum(adjmat0[i,:])==0:
                    orphan_nodes.append(i)            
            if len(orphan_nodes)>0:
                #[0127] if the code goes here there's usually something wrong in the adjmat
                print ('orphan nodes', orphan_nodes)
                #get connected unlabeled (assuming all usgs gages arenot orphans)
                idx_unlabeled_conn = np.array(list(OrderedSet(idx_unlabeled) - OrderedSet(orphan_nodes)))
                #union of connected unlabeled and observations 
                idx_valid = np.array(list(OrderedSet(set(idx_unlabeled_conn).union(idx_obs))))
                #get connected adjmat
                adjmat = adjmat0[idx_valid,:][:,idx_valid]
                print ('adjmat shape', adjmat.shape)
                #we need to renumber the unlabeled and obs 
                idx_valid_unlabel=[]
                idx_valid_obs = []
                for i,item in enumerate(idx_valid):
                    if item in idx_unlabeled_conn:
                        idx_valid_unlabel.append(i)
                    else:
                        idx_valid_obs.append(i)
                assert((len(idx_valid_unlabel)+len(idx_valid_obs))==len(idx_valid))
            else:
                #no orphan nodes
                idx_valid = idx
                idx_valid_unlabel = idx_unlabeled
                idx_valid_obs = idx_obs
                idx_unlabeled_conn = idx_unlabeled

            r_L_train = torch.from_numpy(r_L_train).to(torch.double).to(device)

            X = torch.from_numpy(uMat[:,idx_valid]).to(torch.double).to(device)
            
            yU  = lp_refine(idx_valid_unlabel, idx_valid_obs, r_L_train, X, adjmat, omega=omega, **kwargs)
            uMat[:,idx_unlabeled_conn] =yU.data.cpu().numpy()
            #yU = lp_refine(idx_unlabeled, idx_obs, r_L_train, X, adjmat, omega=omega, **kwargs)
            #uMat[:,idx_unlabeled] = yU.data.cpu().numpy()            
            #use lp refined value to replace the original
            uMat[:,gageNodeIndTrain] = obsData[:,traingageid]
            
            #======plot residual=========
            if isPlotResidual:
                #visualize the difference between ML and NWM due to update
                print ('plotting gage', testgageid[0])
                mydict={
                    'comid': gageCOMId[indx],
                    'node_index':indx,
                    'usgs_gageid':gageDict[gageCOMId[indx]],
                }
                #plot mean absolute error
                plotResidual(args,residualVec=np.mean(np.abs(uMat-nwmMat),axis=0), LOO_gage = mydict)

            #print ('===gage nse after===')
            #newObsData = obsData[:,traingageid]
            #for ix,ind in enumerate(gageNodeIndTrain):
            #    print (ind, Hydrostats.nse(uMat[:,ind],newObsData[:,ix]))
            
            for id1,id2 in zip(testgageid,cv_gageid):
                nseval= Hydrostats.nse(uMat[:,id1],obsData[:,id2])
                maeval= Hydrostats.mae(uMat[:,id1],obsData[:,id2])
                kgeval= Hydrostats.kge_2009(uMat[:,id1],obsData[:,id2])
                corrval=Hydrostats.pearson_r(uMat[:,id1],obsData[:,id2])

                testdict[id1]['model_after']=uMat[:,id1]
                testdict[id1]['nse_after'] = nseval
                testdict[id1]['mae_after'] = maeval 
                testdict[id1]['kge_after'] = kgeval
                testdict[id1]['corr_after'] = corrval

                # if  testdict[id1]['nse_after']>testdict[id1]['nse_before']:
                #     print ('BETTER results after update, %{0:.3f}'.format((testdict[id1]['nse_after']-testdict[id1]['nse_before'])/testdict[id1]['nse_before']*100.0))
                # else:
                #     print ('WORSE results after update, %{0:.3f}'.format((testdict[id1]['nse_after']-testdict[id1]['nse_before'])/testdict[id1]['nse_before']*100.0))

        #print summary
        print (f'===========SUMMARY OMEGA={omega}===========')
        print ('id\tkge_before\tkge_after\tkge_nwm_usgs\tnse_before\tnse_after\tnse_nwm_usgs\tcorr_before\tcorr_after\tcorr_nwm')
        #as0228, change to use metric, can be mae or nse
        metricType = 2 #1, NSE, 2 MAE
        meanMetric = 0.0
        if metricType == 1:
            toler = 5e-4
        elif metricType == 2:
            toler = 1e-2
        else:
            raise Exception('invalid metric type')

        for id in testdict.keys():
            print (id, f"{testdict[id]['kge_before']:.3f}, {testdict[id]['kge_after']:.3f}, {testdict[id]['kge_nwm']:.3f}, | {testdict[id]['nse_before']:.3f},{testdict[id]['nse_after']:.3f}, {testdict[id]['nse_nwm']:.3f} |{testdict[id]['corr_before']:.3f},{testdict[id]['corr_after']:.3f}, {testdict[id]['corr_nwm']:.3f}", )
            if metricType==1:
                metricDiff = testdict[id]['nse_after']-testdict[id]['nse_before']
            elif metricType==2:
                metricDiff = testdict[id]['mae_before']-testdict[id]['mae_after']
            #penalize lower performance
            if metricDiff <0: 
                if metricType==1:
                    metricDiff=100.0*metricDiff
                elif metricType==2:
                    metricDiff=100.0*metricDiff
            meanMetric+= metricDiff
        meanMetric /= len(testdict.keys())
        
        if meanMetric>bestCV_NSE and (meanMetric-prev_nse)>toler:
            print ('*********Best omega found********', meanMetric)
            bestOmega = omega
            bestCV_NSE = meanMetric
        prev_nse = meanMetric

    if not args.tune_omega:
        print ('best omega is ', bestOmega, 'best mean CV KGE ', bestCV_NSE)
        #save testdict
        basestr = f'{args.watershed_name}/{args.modeltag}_lp_seq{args.seq_length}_nwm{args.nwm_ver}'
        if args.uselog:
            basestr+='_uselog'
        if args.addnwm:
            basestr+='_usenwm'
        pkl.dump(testdict, open(f'ensemble_out/{basestr}.pkl', 'wb'))
    else:
        #save testdict
        basestr = f'{args.watershed_name}/{args.modeltag}_lp_seq{args.seq_length}_{args.seed}_nwm{args.nwm_ver}'
        if args.uselog:
            basestr+='_uselog'
        if args.addnwm:
            basestr+='_usenwm'
        pkl.dump(testdict, open(f'ensemble_out/{basestr}.pkl', 'wb'))

def doAll(args,device,adjmat,nwmMat,mlMat,r_L, obsData, gageNodeInd, gageCOMId,gageDict,
            isPlotResidual=False, **kwargs):
    idx = np.arange(adjmat.shape[0])
    idx_unlabeled = np.array(list(OrderedSet(idx) - OrderedSet(gageNodeInd)))
    idx_obs = np.array(gageNodeInd, dtype=np.int16)
    uMat = mlMat.copy()
    orphan_nodes=[]
    for i in range(adjmat.shape[0]):
        if np.sum(adjmat[i,:])==0:
            orphan_nodes.append(i)            
    if len(orphan_nodes)>0:
        #[0127] if the code goes here there's usually something wrong in the adjmat
        print ('orphan nodes', orphan_nodes)
        #get connected unlabeled (assuming all usgs gages arenot orphans)
        idx_unlabeled_conn = np.array(list(OrderedSet(idx_unlabeled) - OrderedSet(orphan_nodes)))
        #union of connected unlabeled and observations 
        idx_valid = np.array(list(OrderedSet(set(idx_unlabeled_conn).union(idx_obs))))
        #get connected adjmat
        adjmat = adjmat[idx_valid,:][:,idx_valid]
        print ('adjmat shape', adjmat.shape)
        #we need to renumber the unlabeled and obs 
        idx_valid_unlabel=[]
        idx_valid_obs = []
        for i,item in enumerate(idx_valid):
            if item in idx_unlabeled_conn:
                idx_valid_unlabel.append(i)
            else:
                idx_valid_obs.append(i)
        assert((len(idx_valid_unlabel)+len(idx_valid_obs))==len(idx_valid))
    else:
        #no orphan nodes
        idx_valid = idx
        idx_valid_unlabel = idx_unlabeled
        idx_valid_obs = idx_obs
        idx_unlabeled_conn = idx_unlabeled

    r_L = torch.from_numpy(r_L).to(torch.double).to(device)

    X = torch.from_numpy(uMat[:,idx_valid]).to(torch.double).to(device)
    
    yU = lp_refine(idx_valid_unlabel, idx_valid_obs, r_L, X, adjmat, omega=args.omega, **kwargs)
    uMat[:,idx_unlabeled_conn] = yU.data.cpu().numpy()
    uMat[:,gageNodeInd] = obsData[:,list(range(len(gageNodeInd)))]
    #yU = lp_refine(idx_unlabeled, idx_obs, r_L_train, X, adjmat, omega=omega, **kwargs)
    #uMat[:,idx_unlabeled] = yU.data.cpu().numpy()

    """
    #use lp refined value to replace the original
    uMat[:,gageNodeIndTrain] = obsData[:,traingageid]

    r_L_train = torch.from_numpy(r_L).to(torch.double).to(device)

    uMat = mlMat.copy()
    X = torch.from_numpy(uMat).to(torch.double).to(device)
    
    yU = lp_refine(idx_unlabeled, idx_obs, r_L_train, X, adjmat, omega=args.omega, **kwargs)
    #use lp refined value to replace the original
    uMat[:,idx_unlabeled] = yU.data.cpu().numpy()
    uMat[:,gageNodeInd] = obsData[:,list(range(len(gageNodeInd)))]
    """
    if isPlotResidual:
        #visualize the difference between ML and NWM due to update
        plotResidual(args,residualVec=np.mean(np.abs(uMat-nwmMat),axis=0), LOO_gage = None)
    return uMat
    
def kalmanFilter(args,device,adjmat,nwmMat,mlMat,comIDdict,gageDict,usgsDict,
                normalize=False, isPlotResidual=False, task=1):
    """data assimilation
    1. task = 1, tune omega
    2. task =2, generate result for fixed omega
    Params
    ------
    args, configuration
    device, device setting
    adjmat, adjacency matrix 
    nwmMat, nwm results at all nodes
    mlMat, ML predictions at all nodes
    comIDdict, map from comID to node index
    gageDict, map from comID to usgs gageid
    usgsDict, dictionary of streamflow DF, key=usgs_gageid
    normalize, True to normalize residual (default False)
    """
    """
    if args.uselog:
        myscaler = pkl.load(open(f'data/{args.watershed_name}/myscaler_log.pkl', 'rb'))  
    else:
        myscaler = pkl.load(open(f'data/{args.watershed_name}/myscaler.pkl', 'rb'))  
    """
    #remove self loop in adjmat
    np.fill_diagonal(adjmat,0.0)

    #check adjmat for bad nodes
    diagnoseAdjMat(args,adjmat, comIDdict)
    
    #create a map from gage comid to node index
    gageNodeInd=[]
    #create a map from node index to comID
    gageCOMId = list(gageDict.keys())

    #nTest is total data points used for testing
    nTest = mlMat.shape[0]
        
    for item in gageDict.keys():
        try:
            #do reverse lookup to find col index in nwmdf
            gageNodeInd.append(comIDdict[item])
        except:
            pass    

    gageNodeInd=np.array(gageNodeInd)        
    sortedind = np.argsort(gageNodeInd)
    print ('gageNodeInd', gageNodeInd)
    
    #Concatenate usgs obs data into DF
    bigDF = [usgsDict[gageDict[item]] for item in gageDict.keys()]     
    #will this filter out nan values?   
    bigDF = pd.concat(bigDF, axis=1) 
    print ('usgsDF has missing data', bigDF.isnull().values.any())
    
    obsData = bigDF.to_numpy()
    if task==1:
        #use observation data with offset to match training setup
        obsData = obsData[args.seq_length:args.seq_length+nTest,:] #use offset on training data
    else:
        obsData = obsData[-nTest:,:] #truncate to testing period only
    #test for nan
    print ('Obsdata has null values ', np.isnan(obsData).any())
    
    if np.isnan(obsData).any():
        goodind = np.where(~np.isnan(obsData).any(axis=1))[0]
        obsData = obsData[goodind,:]
        mlMat = mlMat[goodind,:]
        nwmMat = nwmMat[goodind,:]
        print ('truncated obs,ml,nwm', obsData.shape, mlMat.shape,nwmMat.shape)

    #sort according to column order
    gageNodeInd = gageNodeInd[sortedind].tolist() #sort in ascending col order
    gageCOMId = np.array(gageCOMId)[sortedind].tolist()        
    print ('sorted gageNodeInd', gageNodeInd)
    print ('sorted gageCOMID', gageCOMId)
    obsData = obsData[:,sortedind] #sort in ascending col order

    print ('===Original GAGE KGE Values before LOO ===')
    print ('node      gagecomid      usgsgageid     nwm_usgs   ml_usgs ')
    for ix,ind in enumerate(gageNodeInd):
        #print NSE between ML and Obs
        print (ind, gageCOMId[ix], gageDict[gageCOMId[ix]], 
            '{:.3f}'.format(Hydrostats.kge_2009(nwmMat[:,ind],obsData[:,ix])),
            '{:.3f}'.format(Hydrostats.kge_2009(mlMat[:,ind],obsData[:,ix])),
        )

    #calculate ML and obs residual
    r_L = obsData - mlMat[:,gageNodeInd]     

    #initialize with false
    kwargs={'isNormalize':False}
    if normalize:
        r_L_mean = np.mean(r_L)
        r_L_std = np.std(r_L)
        r_L = (r_L - r_L_mean)/r_L_std
        kwargs['mean'] = r_L_mean
        kwargs['std'] = r_L_std
        kwargs['isNormalize'] = True
        """        
        qt = QuantileTransformer(n_quantiles=50, random_state=0, output_distribution='normal')
        print ('rL', r_L.shape)
        r_L = qt.fit_transform(r_L)
        kwargs['isNormalize'] = True
        kwargs['transformer'] = qt
        """
    #test if there's invalid value
    print ('Test if r_L has nan: ', np.isnan(r_L).any())
    if task==1:
        #do cv to find omega
        args.tune_omega=True
        doLOO(args,device, adjmat,nwmMat,mlMat,r_L,obsData,gageNodeInd, gageCOMId,gageDict,
            isPlotResidual, **kwargs)
    elif task==2:
        args.tune_omega=False
        #do cv using a fixed omega
        doLOO(args,device, adjmat,nwmMat,mlMat,r_L,obsData,gageNodeInd, gageCOMId,gageDict,
            isPlotResidual, **kwargs)
        #use all gages for label propagation
        uMat = doAll(args,device, adjmat,nwmMat,mlMat,r_L,obsData,gageNodeInd, gageCOMId,gageDict,
            isPlotResidual, **kwargs)
        return uMat
    else:
        raise NotImplementedError("Not implemented")

def plotResidual(args,residualVec,usercmap="rainbow", LOO_gage=None):
    """Visualize node difference due to label propagation
    """
    from nwmutils import plotNetworkResidual
    #hard coding!
    nhdplusdbroot = '/work2/02248/alexsund/maverick2/nwm/nhdplusdb'
    wobj = loadWatershed(args, dbroot=nhdplusdbroot, returnWobj=True)
    comIDset,comIDdict,allDF,goodInd,gageDict = loadWatershed(args, dbroot=nhdplusdbroot, returnWobj=False)
    rootdir = wobj.rootdir
    #remove bad values
    residualVec[np.isinf(residualVec)]=0.0
    print ('min residual', np.min(residualVec))
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
        modeltag=args.modeltag
    )

def diagnoseAdjMat(args, adjmat, comIDdict):
    """[01272022] use this function to check orphaned nodes in adjmat
    """
    #check adjmat
    nwm_ver = args.nwm_ver
    nwmfile = f'nwmdfv20_{args.interval}.pkl' if nwm_ver=='2.0' else f'nwmdfv21_{args.interval}.pkl'
    nwmdf = pkl.load(open('data/{0}/{1}'.format(args.watershed_name, nwmfile),'rb'))
    #create inverse map from node index to comid
    nodeBig={}
    for ix,comid in enumerate(nwmdf.columns):
        nodeBig[ix] = comid
    print ('total nodes =', len(nodeBig))
    nodeDict={}
    for key,val in comIDdict.items():
        nodeDict[val] = key
    if nwm_ver=='2.0':
        adjfile = 'data/{0}/adjinfo.pkl'.format(args.watershed_name)
    else:
        adjfile = 'data/{0}/adjinfo{1}.pkl'.format(args.watershed_name, nwm_ver)
    #note the following info is on the whole node set. It's generated by genrivernet.py
    resDict = pkl.load(open(adjfile, 'rb'))
    startNode = resDict['start_node'] 
    endNode   = resDict['end_node']
    
    nbadnode=0
    for i in range(adjmat.shape[0]):
        if np.sum(adjmat[i,:])==0:
            comid = nodeDict[i]
            nbadnode+=1
            for ix,j in enumerate(startNode):
                if nodeBig[j]==comid:
                    print ('orphan node', comid, nodeBig[endNode[ix]])

    if nbadnode>0:
        print (f'Warning, {nbadnode} bad nodes found in the adj mat')