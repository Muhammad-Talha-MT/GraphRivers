#author: alex sun
#date: 06072023
#purpose: extract input/output data for erhu
#From Xiaowei: 
# input driver matrix of [N(# segments), T(# time steps), D(input feature dimension)], 
# NWM model outputs [N(# segments), T(# time steps)], 
# the real observations [N(# segments), T(# time steps)], and an adjacency matrix [N(# segments), N(# segments)]. 
# If these data are stored in pkl files he can extract them. 
# It would be great if you could send 3H data so he can quickly run some experiments and compare with your results. 
#============================================================================================================================================
import os
import pickle as pkl
from util_gtnet import load_config
from loadrivernet import formAdjacencyMat,loadNWMDF,getUSGSData
import pandas as pd

def getAdj(args, goodInd, comIDdict):
    """Retrieve adj matrix
    Params
    ------
    args: configuration yml 
    goodInd: valid comids to be used in the network
    comIDdict: mapping between valid comid and node index, node indices start from 0 to nnode
    """
    A = formAdjacencyMat(
        args.rootdir,
        args.watershed_name,
        goodInd,
        comIDdict,
        isWeighting=args.network.weighted_adj,
        isDirected=args.network.directed_adj,
        connect_exta_path=False,
        add_self_loop=True, 
        nwm_ver=args.data.nwm_ver )

    return A

def getForcing(args, nnode):
    """Retrieve forcing data
    Params
    ------
    nnode, number of nodes
    """
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
    else:
        raise ValueError("invalid forcing source")

    print (forcingDF.keys())
    return forcingDF

def getObsData(args, gageDict, nwmDF):
    start_date = nwmDF.index[0].strftime('%Y-%m-%d')
    end_date =   nwmDF.index[-1].strftime('%Y-%m-%d')
    if args.data.interval == '3H':
        usgsDict = getUSGSData(args.watershed_name,gageDict.values(), 
                           startDate=start_date, endDate=end_date,
                           reLoad=args.data.reload_usgs, valueType='iv')
    else:
        usgsDict = getUSGSData(args.watershed_name,gageDict.values(),
                           startDate=start_date, endDate=end_date,
                           reLoad=args.data.reload_usgs, valueType='dv')

    return usgsDict

def main(config_file):
    config = load_config(config_file)
    #
    comIDset,comIDdict,nwmdf,goodInd,gageDict = loadNWMDF(config)    
    A =getAdj(config, goodInd, comIDdict)
    nnode = A.shape[0]
    foringDF = getForcing(config, nnode)
    print (nwmdf.shape)
    print (foringDF[comIDset[0]].shape)

    usgsDict = getObsData(config, gageDict, nwmdf)
    print (usgsDict)

    allData = {'adj': A, 'comIDdict': comIDdict, 'nwm': nwmdf, 'forcing': foringDF, 'qobs': usgsDict}
    pkl.dump(allData, open(f'data/{config.watershed_name}/{config.watershed_name}_alldata.pkl', 'wb'))

if __name__ == "__main__":
    main(config_file='config_houston.yaml')