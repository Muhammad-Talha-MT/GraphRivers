#author: alex sun
#date: 02/02/2021
#desc: utility functions
#rev: 04262021, clean up for github
#=====================================================================
#third party packages
import matplotlib.pyplot as plt
import pandas as pd
import hydrostats as Hydrostats
import os,sys
import geopandas as gpd
import dataretrieval.nwis as nwis
from  datetime import date, datetime
import pickle as pkl
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap,TwoSlopeNorm

#from this repo
from nwmrenyi import NWMSubsetRenci

def readShp(filename, comIDColName):
    """Read shp file containing nhdflow flowline information
    comIDName, field name corresponding to comID
    """
    import fiona
    shpfile = os.path.abspath(filename)
    comIDset=[]
    with fiona.open(shpfile, 'r') as src:
        for feat in src:
            comIDset.append(feat['properties'][comIDColName])

    return comIDset

def readPPShp(filename, comIDColName):
    """Read extended flowline with downstream huc info in it
    Returns
    -------
    comIDset, pandas series
    downHUCset, pandas series
    """
    shpfile = os.path.abspath(filename)
    gdf = gpd.read_file(shpfile)    
    comIDset = gdf[comIDColName]
    HUCset = gdf['HUC_12']
    downHUCset = gdf['DOWNHUC']
    return comIDset,HUCset,downHUCset

def plotNetwork(watershedname, prettyname, flowlineshp, comIDset, gagelocshp=None, 
                gageSet=None, comIDColName='COMID',floIDColName='FLComID',
                basinboundshp=None,resDict=None):
    """Plot river network    
    Parameters
    ----------
    watershedname, watershed named used in output file
    flowlineshp, nhdplus flowline shapefile
    comIDset, a subset of comid to be used in constructing the rivernet
    gagelocshp, nhdplus gageloc shapefile [see E:\CAMELS\hru_1423000.mxd]
    gageSet, subset of gage locs used in label propagation
    comIDColName: name of the comid column in shapfile    
    boundshp, full path to the watearshed polygon file
    """
    gdf = gpd.read_file(flowlineshp)
    gdfsub = gdf[gdf[comIDColName].isin(comIDset)]

    nnodes = gdfsub.shape[0]
    print ('in plotting, number of comids ', nnodes)
    fig,ax=plt.subplots(1,1,figsize=(12,8))
        
    gdf.plot( ax=ax, edgecolor='gray', linewidth=0.5, alpha=0.5, legend="nhd+")
    #gdfsub.plot(ax=ax,edgecolor='#909497', linewidth=1.0, marker='o' )
    #as01032020, switch to light blue
    gdfsub.plot(ax=ax,edgecolor='#5DADE2', linewidth=1.0, marker='o' )
    gdfsub['points'] = gdfsub.apply(lambda x: [y for y in x['geometry'].coords], axis=1)

    #as02082021 plot edge node
    for index,row in gdfsub.iterrows():
        #print the from node
        ax.plot(row['points'][0][0],row['points'][0][1],'o', markerfacecolor='gray',
                    markeredgecolor='gray', markersize=3)
    #as01132022, plot extra edges
    if not resDict is None:
        for item in resDict['extra_path']:
            #plot extra edge if both start/end nodes are good
            if item[0] in comIDset and item[1] in comIDset:
                #return all points in a reach, use the last point for the end point
                pt1 = gdfsub[gdfsub[comIDColName]==item[0]]['points'].tolist()[0][-1]
                pt2 = gdfsub[gdfsub[comIDColName]==item[1]]['points'].tolist()[0][-1]
                #print (pt1, pt2)
                x0,y0=pt1[0:2]
                x1,y1=pt2[0:2]
                ax.plot([x0,x1],[y0,y1], linewidth=1.5, linestyle=':', color='black')                
                
    if not gagelocshp is None:
        gdfgage = gpd.read_file(gagelocshp)
        for index, row in gdfgage.iterrows():
            if not gageSet is None and row[floIDColName] in gageSet:
                gdfgage.loc[[index], 'geometry'].plot(ax=ax, color='red', markersize=40)
            #as02082021 comment this out to avoid confusion, only label the good gages
            #else:
            #    gdfgage.loc[[index], 'geometry'].plot(ax=ax, color='red',markersize=40)
    
    if not basinboundshp is None:
        #add basin boundary
        boundgdf = gpd.read_file(basinboundshp)
        boundgdf = boundgdf.to_crs('EPSG:4326')
        boundgdf.boundary.plot(ax=ax, alpha=0.8, edgecolor='#626567')

    plt.title(f'Watershed: {prettyname},#Reaches={nnodes}')

    plt.savefig('outputs/rivernet{0}.png'.format(watershedname))
    plt.close()

def plotFPPNetwork(watershedname, prettyname, flowlineshp, comIDset, h12shp,
                gagelocshp=None, gageSet=None, comIDColName='COMID',floIDColName='FLComID',
                basinboundshp=None,resDict=None):
    """Plot H12 subbasins and pour points
    Parameters
    ----------
    watershedname, watershed named used in output file
    flowlineshp, nhdplus flowline shapefile
    comIDset, a subset of comid to be used in constructing the rivernet
    gagelocshp, nhdplus gageloc shapefile [see E:\CAMELS\hru_1423000.mxd]
    gageSet, subset of gage locs used in label propagation
    comIDColName: name of the comid column in shapfile    
    boundshp, full path to the watearshed polygon file
    """
    #plot subbasins in the background
    
    gdf = gpd.read_file(flowlineshp)
    gdfsub = gdf[gdf[comIDColName].isin(comIDset)]

    nnodes = gdfsub.shape[0]
    print ('In plotting fpp network, number of comids ', nnodes)
    
    fig,ax=plt.subplots(1,1,figsize=(12,8))    
    if not h12shp is None:
        gdf12 = gpd.read_file(h12shp)
        gdf12.boundary.plot(ax=ax,edgecolor='#5DADE2', linewidth=0.5, alpha=0.5)

    #plot pour points    
    gdfsub.plot(marker='o', color='gray',edgecolor='gray', markersize=8, ax=ax)
    
    if not basinboundshp is None:
        #add basin boundary
        boundgdf = gpd.read_file(basinboundshp)
        boundgdf.boundary.plot(ax=ax, alpha=0.8, edgecolor='#626567')

    plt.title(f'Watershed: {prettyname},#HUC12 Basins={nnodes}')

    plt.savefig('outputs/rivernet{0}.png'.format(watershedname))
    plt.close()


def plotNetworkNSE(watershedname, prettyname, flowlineshp, comIDset, nseArr, gagelocshp=None, 
                gageSet=None, comIDColName='COMID',floIDColName='FLComID',
                colorbar=True, usercmap='rainbow', vmin=-1,vmax=1,mlmodel='gwnet',plotNSECDF=False, 
                specialNodes=None, basinboundshp=None, observable_comidset=None, 
                metricName='nse', nwm_ver='2.0', usenwm=True, removeAxis=False, 
                outputeps=False, isResidual=False, exportShpFile=False,
                forcing_source=None, interval=None):
    if plotNSECDF:
        from statsmodels.distributions.empirical_distribution import ECDF
    """Plot river network NSE (ML Model vs. NWM)
    Parameters
    ----------
    watershedname, watershed named used in output file
    prettyname, beautiful name for printing
    flowlineshp, nhdplus flowline shapefile
    comIDset, a subset of comid to be used in constructing the rivernet
    nseArr, array of all NSE values
    gagelocshp, nhdplus gageloc shapefile [see E:\CAMELS\hru_1423000.mxd]
    gageSet, subset of gage locs used in label propagation
    comIDColName: name of the comid column in shapfile
    mlmodel: specify model type (only mtgnn or gwnet is allowed)
    plotNSECDF: true to plot EDF and map plot side by side
    """
    gdf = gpd.read_file(flowlineshp)
    print ('in plotting NSE, ', comIDColName)
    gdfsub = gdf[gdf[comIDColName].isin(comIDset)]
    nnodes = gdfsub.shape[0]
    print ('in plotting, number of comids ', nnodes)
    if plotNSECDF:
        fig = plt.figure(figsize=(10,10),dpi=350)
        gs = fig.add_gridspec(2, 2, width_ratios=(1,4),height_ratios=(1,5),
                    left=0.05, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.0, hspace=0.05)            
        ax0 = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[:, 1])
    else:
        fig,ax=plt.subplots(1,1,figsize=(12,8))
        
    gdf.plot(ax=ax, edgecolor='gray', linewidth=0.7, alpha=0.5, legend="nhd+")
    #gdfsub.plot(ax=ax,edgecolor='#909497', linewidth=1.0, marker='o' )
    gdfsub.plot(ax=ax,edgecolor='#5F6A6A', linewidth=1.2, marker='o' )
    #Explode muti-part geometries into multiple single geometries.
    gdfsub = gdfsub.explode()
    gdfsub['points'] = gdfsub.apply(lambda x: [y for y in x['geometry'].coords], axis=1)

    #as02082021 plot edge node
    #as06032021 loop according to the order of comIDset
    if isResidual:
        divnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)        
        cmap = plt.get_cmap('coolwarm')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=divnorm)
    else:
        cmap = plt.get_cmap(usercmap)
        #min_nse = np.max([-1, np.min(nseArr)])
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for ix,item in enumerate(comIDset):
        nodecolor = sm.to_rgba(nseArr[ix])
        if not specialNodes is None and ix in specialNodes:
            markeredgecolor = '#6495ED'
            markersize=6
        else:
            markeredgecolor = nodecolor
            markersize=3
            
        row = gdfsub[gdfsub[comIDColName] == item].iloc[0,:]
        #as01282022, add special treatment for observable set
        if not observable_comidset is None and  item in observable_comidset:
            ax.plot(row['points'][0][0],row['points'][0][1],'s', 
                        markerfacecolor=nodecolor,
                        markeredgecolor='#909497',  #gray edge
                        markersize=markersize+5)
        else:
            ax.plot(row['points'][0][0],row['points'][0][1],'o', 
                        markerfacecolor=nodecolor,
                        markeredgecolor=markeredgecolor, 
                        markersize=markersize+1)
    if exportShpFile:
        #asun 10072022
        df = pd.DataFrame(np.c_[np.array(comIDset, dtype=np.int32), nseArr])
        df.columns = [comIDColName, 'NSE']
        print (df)
        print (df.shape)
        #
        gdfout = gdfsub.merge(df, on=comIDColName)
        gdfout = gdfout[[comIDColName, 'geometry', 'NSE']]
        print (gdfout.head(5))
        print (gdfout.columns)
        print ('saving to shapefile...')
        if forcing_source is None:
            #for backward compatibility
            gdfout.to_file(f"outputs/{watershedname}/{metricName}_rivernet{watershedname}_{mlmodel}_node{nnodes}_nwm{nwm_ver}.shp")
        else:
            gdfout.to_file(f"outputs/{watershedname}/{metricName}_rivernet{watershedname}_{mlmodel}_node{nnodes}_nwm{nwm_ver}_{forcing_source}_{interval}.shp")

    if metricName=='nse':
        ax.set_title(f'Watershed: {prettyname},#Reaches={nnodes},medianNSE={np.median(nseArr):3f}')
    else:
        ax.set_title(f'Watershed: {prettyname}, Network nodes,{nnodes}')

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%",pad=0.0)            
        if isResidual:
            cax.set_title(f'${{\Delta}}$ {metricName.upper()}')
            plt.colorbar(sm,ax=ax,cax=cax,shrink=0.8,extend='max')
        else:
            cax.set_title(metricName.upper())
            plt.colorbar(sm,ax=ax,cax=cax,shrink=0.8)

    if not gagelocshp is None:
        gdfgage = gpd.read_file(gagelocshp)
        for index, row in gdfgage.iterrows():
            if not gageSet is None and row[floIDColName] in gageSet:
                gdfgage.loc[[index], 'geometry'].plot(ax=ax, color='none', edgecolor='black', markersize=50)            
            #else:
            #    #gdfgage.loc[[index], 'geometry'].plot(ax=ax, color='red',edgecolor='black', markersize=40)
            #    print ('bypassed gage not in set',row[floIDColName] )


    if not basinboundshp is None:
        #add basin boundary
        boundgdf = gpd.read_file(basinboundshp)
        boundgdf.boundary.plot(ax=ax, alpha=0.8, linewidth=0.5, edgecolor='k', cmap=cmap)

    if plotNSECDF:
        ecdf = ECDF(nseArr, side='right')
        ax0.plot(ecdf.x,ecdf.y, linestyle="-",
            color="steelblue",label="ML")
        ax0.set_ylabel('ECDF')
        if isResidual:
            ax0.set_xlabel(f'${{\Delta}}$ {metricName.upper()}')
        else:
            ax0.set_xlabel(metricName.upper())

    if removeAxis:
        ax.set_axis_off()

    #create an output folder if it does not exist
    os.makedirs(f'outputs/{watershedname}', exist_ok=True)
    usenwmstr='_usenwm' if usenwm else ''
    if isResidual:
        metricName=metricName+'_residual'
    if outputeps:
        if forcing_source is None:
            plt.savefig(f'outputs/{watershedname}/{metricName}_rivernet{watershedname}_{mlmodel}_node{nnodes}_nwm{nwm_ver}{usenwmstr}.eps')        
        else:
            plt.savefig(f'outputs/{watershedname}/{metricName}_rivernet{watershedname}_{mlmodel}_node{nnodes}_nwm{nwm_ver}{usenwmstr}_{forcing_source}_{interval}.eps')        
    else:
        if forcing_source is None:
            plt.savefig(f'outputs/{watershedname}/{metricName}_rivernet{watershedname}_{mlmodel}_node{nnodes}_nwm{nwm_ver}{usenwmstr}.png')
        else:
            plt.savefig(f'outputs/{watershedname}/{metricName}_rivernet{watershedname}_{mlmodel}_node{nnodes}_nwm{nwm_ver}{usenwmstr}_{forcing_source}_{interval}.png')

    plt.close()


def getNWISData(gageid, startDate='1993-01-01',endDate='2018-12-31', valueType='dv'):
    """get usgs gage data
    Parameters
    ---------
    gageid: USGS stream gage id
    Note: USGS REST parameter
    instantaneous values (iv)
    daily values (dv)
    statistics (stat)
    site info (site)
    discharge peaks (peaks)
    discharge measurements (measurements)
    """
    # specify the USGS site code for which we want data.
    site = gageid
    # get usgs data    
    df = nwis.get_record(sites=site, service=valueType, start=startDate, end=endDate)

    # get basic info about the site
    #df3 = nwis.get_record(sites=site, service='site')    
    print (df.columns)
    if valueType == 'dv':
        df = df[['00060_Mean']]
    elif valueType == 'iv':
        df = df[['00060']]

    df.columns=['Q']
    print (df.head(5))
    return df

def compareNWMUSGS(watershed,comID,readData=False, gageID=None, gageDF=None,conversion=True):
    """compare nwm to usgs data
    Parameters
    ---------
    watershed: watershed name
    comID: nhd+ comid
    readData: if True read from disc, otherwise use gageDF values
    gageID, usgs gage ID
    gageDF, dataframe containing streamflow data
    conversion, True to convert from cfs to m3/s

    Returns
    -------
    residual at the site
    """
    cfs2m3s = 0.028316847    
    if readData:
        #this reads "cleaned" data
        filename = 'data/{0}/{1}.txt'.format(watershed, gageID)
        #usgs streamflow in cfs        
        df = pd.read_csv(filename, sep="\t", usecols=[2,3], header=None)
        df = pd.DataFrame(df.iloc[:,1].to_numpy(), index=df.iloc[:,0])
        df.columns=['Q']

    else:
        df = gageDF
    
    if conversion:        
        df['Q'] = df['Q'].apply(lambda x: x*cfs2m3s)
    df = df.dropna()
    #NWM in m3/s
    comIDset = list([comID])
    a = NWMSubsetRenci(comIDset, yearRng=[1993,2018])
    a.getData()
    nwmdf = a.reSample(a.df,rule='24H')
    #align the time axis, assuming df may have missing data
    nwmdf.index = pd.to_datetime(nwmdf.index,utc=True)
    df.index = pd.to_datetime(df.index,utc=True)
    mergedDF = df.join(nwmdf)


    fig,ax = plt.subplots(1,1)
    ax.plot_date(mergedDF.index, mergedDF.iloc[:,0], '-', xdate=True, color='gray', alpha=0.7,label='Obs')
    ax.plot_date(mergedDF.index, mergedDF.iloc[:,1],'-', xdate=True, color='coral', label='NWM')
    ax.set_yscale('log')
    ax.set_xlabel('Date')
    ax.set_ylabel('Q (m3/s)')
    plt.grid(True)
    nse = Hydrostats.nse(mergedDF.iloc[:,1], mergedDF.iloc[:,0])
    corr = Hydrostats.pearson_r(mergedDF.iloc[:,0], mergedDF.iloc[:,1])

    plt.title(f'Gage {gageID}, NSE={nse:.3f},Corr={corr:.3f}')
    plt.legend()

    plt.savefig('data/{0}/plt_{0}_{1}.png'.format(watershed,gageID))
    plt.close()

    return (mergedDF.iloc[:,0].to_numpy()-mergedDF.iloc[:,1].to_numpy())

def comparePRMS2USGS(watershed,readData=False, gageID=None, gage2prmsDict=None):
    """compare nwm to usgs data
    Parameters
    ---------
    watershed: watershed name
    comID: nhd+ comid
    readData: if True read from disc, otherwise use gageDF values

    """
    cfs2m3s = 0.028316847    
    if readData:
        #this reads "cleaned" usgs data
        filename = 'data/{0}/{1}.txt'.format(watershed, gageID)
        #usgs streamflow in cfs        
        df = pd.read_csv(filename, sep="\t", usecols=[2,3], header=None)
        df = pd.DataFrame(df.iloc[:,1].to_numpy(), index=df.iloc[:,0])
        df.columns=['Q']

    else:
        df = gageDF #what is this, not defined

    #df['Q'] = df['Q'].apply(lambda x: x*cfs2m3s)
    df = df.dropna()
    
    #get PRMS
    filename = f'data/{watershed}/nsegment_seg_outflow.csv'
    #'09112200','09107000','09109000','09110000','09112500']
    #mapping between gage list and HRU segment list zero-index
    print ('colid ', gage2prmsDict[gageID])
    dfPRMS = pd.read_csv(filename, usecols=[0,gage2prmsDict[gageID]], header=0, parse_dates=[0],index_col=0)
    dfPRMS.columns=['Qprms']
    #dfPRMS['Qprms'] = dfPRMS['Qprms'].apply(lambda x: x*cfs2m3s)
    #align the time axis, assuming df may have missing data
    dfPRMS.index = pd.to_datetime(dfPRMS.index,utc=True)
    df.index = pd.to_datetime(df.index,utc=True)
    mergedDF = df.join(dfPRMS)
    #print (mergedDF.head(20))

    fig,ax = plt.subplots(1,1)
    ax.plot_date(mergedDF.index, mergedDF.iloc[:,0], '-', xdate=True, color='gray', alpha=0.7,label='Obs')
    ax.plot_date(mergedDF.index, mergedDF.iloc[:,1],'-', xdate=True, color='coral', label='NWM')
    ax.set_yscale('log')
    ax.set_xlabel('Date')
    ax.set_ylabel('Q (cfs)')
    plt.grid(True)
    nse = Hydrostats.nse(mergedDF.iloc[:,0], mergedDF.iloc[:,1])
    corr = Hydrostats.pearson_r(mergedDF.iloc[:,0], mergedDF.iloc[:,1])

    plt.title(f'Gage {gageID}, NSE={nse:.3f},Corr={corr:.3f}')
    plt.legend()

    plt.savefig('data/{0}/plt_{0}_{1}_PRMS.png'.format(watershed,gageID))
    plt.close()

def checkUSGSGauge(gageshpfile, startDate='2000-01-01', endDate='2018-12-31'):
    """Check the length of records of usgs streamflow gages
    This method helps to select gages with full records during the specified period
    
    Params
    ------
    gageshpfile, name of the gageinfo shape file (must be extracted from nhdplusgageinfo)
    startDate, endDate, duration of the period
    """
    #get usgs gage id
    gdfgage = gpd.read_file(gageshpfile)

    totaldays = (datetime.strptime(endDate, '%Y-%m-%d') - datetime.strptime(startDate,'%Y-%m-%d')).days
    goodSites=[]
    goodFComIDs=[]
    for index, row in gdfgage.iterrows():
        # specify the USGS site code for which we want data.
        site = (row['SOURCE_FEA'])
        # get daily values (dv)
        res = nwis.get_record(sites=site, service='dv', start=startDate, end=endDate)
        recordDays = res.shape[0]
        print ('usgs id', site, ' comid', row['FLComID'], '# data', recordDays)
        if recordDays>=totaldays:
            goodSites.append("{0}".format(site))
            goodFComIDs.append(row['FLComID'])
    print (goodSites)
    print (goodFComIDs)
    return (goodSites, goodFComIDs)

def getUSGSData(watershed, gageIDset, startDate='1993-01-01', endDate='2018-12-31',reLoad=False, 
                valueType='dv'):
    """Get usgs streamflow gages data and persist it
    
    Params
    ------
    gageIDset, a set of USGS gage id
    startDate,endDate, duration of the period
    reLoad, True to retrieve the data again
    valueType, type of data variable used by USGS web service
    Returns
    -------
    usgsDataDict, dictionary of usgs streamflow data
    """
    totaldays = (datetime.strptime(endDate, '%Y-%m-%d') - datetime.strptime(startDate,'%Y-%m-%d')).days

    if reLoad:
        cfs2m3s = 0.028316847    

        usgsDataDict={}
        for site in gageIDset: 
            print (site)
            # specify the USGS site code for which we want data.        
            df = getNWISData(gageid=site, startDate=startDate,endDate=endDate, valueType= valueType)
            #!!!!!!convert to m3/s to match with nwm
            df['Q'] = df['Q']*cfs2m3s   
            recordDays = df.shape[0]
            print ('site', site, ' record days', recordDays, 'total days ', totaldays)
            usgsDataDict[site]=df
        pkl.dump(usgsDataDict, open('data/{0}/usgs_data_{1}.pkl'.format(watershed, valueType), 'wb'))
    else:
        usgsDataDict = pkl.load(open('data/{0}/usgs_data_{1}.pkl'.format(watershed, valueType), 'rb'))
        #do clean up for 3H
        if valueType == 'iv':
            #aggregate data to 3H
            for key, df in usgsDataDict.items():
                #print ('before resampling ', df.head(10))                
                df.index = pd.to_datetime(df.index,utc=True)    
                df = df.resample('3H').mean()
                #print ('after resampling ', df.head(10))
                mask =  (df.index >= pd.to_datetime(startDate).tz_localize('UTC')) & (df.index <= pd.to_datetime(endDate).tz_localize('UTC'))
                df = df.loc[mask]
                usgsDataDict[key] = df
                #print ('df shape', df.shape, 'total 3H should be', totaldays*8)

    return usgsDataDict


def plotNetworkResidual(watershedname, prettyname, flowlineshp, comIDset, residualArr, gagelocshp=None, 
                gageSet=None, comIDColName='COMID',floIDColName='FLComID',
                colorbar=True, usercmap='rainbow', LOO_gage=None, basinboundshp=None,addnwm=True,
                uselog=False, connect_exta_path=False, nwm_ver='2.0', removeAxis = True,
                vmin = None, vmax= None, modeltag='gwn'):
    """Plot residuals (diff between gcn and nwm) due to label propagation over the river network 
    Parameters
    ----------
    watershedname, watershed named used in output file
    prettyname, beautiful name for printing
    flowlineshp, nhdplus flowline shapefile
    comIDset, a subset of comid to be used in constructing the rivernet
    nseArr, array of all NSE values
    gagelocshp, nhdplus gageloc shapefile [see E:\CAMELS\hru_1423000.mxd]
    gageSet, subset of gage locs used in label propagation
    comIDColName: name of the comid column in shapfile
    """
    gdf = gpd.read_file(flowlineshp)
    print ('in plotting NSE, ', comIDColName)
    gdfsub = gdf[gdf[comIDColName].isin(comIDset)]
    nnodes = gdfsub.shape[0]
    print ('in plotting, number of comids ', nnodes)
    fig,ax=plt.subplots(1,1,figsize=(12,8))
        
    gdf.plot(ax=ax, edgecolor='gray', linewidth=0.7, alpha=0.5, legend="nhd+")
    gdfsub.plot(ax=ax,edgecolor='#909497', linewidth=1.2, marker='o' )
    gdfsub['points'] = gdfsub.apply(lambda x: [y for y in x['geometry'].coords], axis=1)

    #as02082021 plot edge node
    #as06032021 loop according to the order of comIDset
    #cmap = plt.get_cmap(usercmap)
    #as03182022, change to turbomap for residual plot
    #turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]
    #cmap=ListedColormap(turbo_colormap_data)
    cmap = plt.get_cmap('viridis_r')
    if vmin is None:
        vmin = np.min(residualArr)
    if vmax is None:
        vmax = np.max(residualArr)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for ix,item in enumerate(comIDset):
        nodecolor = sm.to_rgba(residualArr[ix])
        markeredgecolor = nodecolor
            
        row = gdfsub[gdfsub[comIDColName] == item].iloc[0,:]
        ax.plot(row['points'][0][0],row['points'][0][1],'o', 
                    markerfacecolor=nodecolor,
                    markeredgecolor=markeredgecolor, 
                    markersize=3)
    if not LOO_gage is None:
        ax.set_title(f'Watershed: {prettyname},#Reaches={nnodes}, LOO gage {LOO_gage["usgs_gageid"]}')
    else:
        ax.set_title(f'Watershed: {prettyname},#Reaches={nnodes}, All gages')
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%",pad=0.0)            
        cax.set_title("Residual\n[m3/s]")
        plt.colorbar(sm,ax=ax,cax=cax)

    if not gagelocshp is None:
        gdfgage = gpd.read_file(gagelocshp)
        for index, row in gdfgage.iterrows():
            if not gageSet is None and row[floIDColName] in gageSet:
                if not LOO_gage is None:
                    if row[floIDColName]==LOO_gage['comid']:
                        gdfgage.loc[[index], 'geometry'].plot(ax=ax, color='none', edgecolor='black', markersize=50)            
                        ax.annotate(LOO_gage['usgs_gageid'], xy=(gdfgage.loc[[index], 'geometry'].x, gdfgage.loc[[index], 'geometry'].y), xytext=(3, 3), textcoords="offset points")
                else:
                    gdfgage.loc[[index], 'geometry'].plot(ax=ax, color='none', edgecolor='black')#, makersize=50)           
    
    if not basinboundshp is None:
        #add basin boundary
        boundgdf = gpd.read_file(basinboundshp)
        boundgdf.boundary.plot(ax=ax, alpha=0.8, edgecolor='black',cmap=cmap)
    #remove lat/lon
    if removeAxis:
        ax.set_axis_off()

    basestr = f"{modeltag}_residual_rivernet_{watershedname}_node{len(comIDset)}_nwm{nwm_ver}"
    if addnwm:
        basestr += '_addnwm'
    if not LOO_gage is None:
        basestr += f"_{LOO_gage['usgs_gageid']}"        
    if uselog: 
        basestr += "_log"
    if connect_exta_path:
        basestr += "_extrapath"
    #as0318, change this to eps when plotting Figure 9
    #plt.savefig('outputs/{0}/{1}.eps'.format(watershedname, basestr))    
    plt.savefig('outputs/{0}/{1}.png'.format(watershedname, basestr))
    plt.close()

def plotAdjNetwork(watershedname, prettyname, flowlineshp, comIDset, comdIDdict, adj,
                gagelocshp=None, gageSet=None, comIDColName='COMID',
                floIDColName='FLComID',basinboundshp=None,resDict=None):
    """Plot river network adj matrix    
    Parameters
    ----------
    watershedname, watershed named used in output file
    flowlineshp, nhdplus flowline shapefile
    comIDset, a subset of comid to be used in constructing the rivernet
    gagelocshp, nhdplus gageloc shapefile [see E:\CAMELS\hru_1423000.mxd]
    gageSet, subset of gage locs used in label propagation
    comIDColName: name of the comid column in shapfile    
    boundshp, full path to the watearshed polygon file
    """
    gdf = gpd.read_file(flowlineshp)
    print ('in plotting, ', comIDColName)
    gdfsub = gdf[gdf[comIDColName].isin(comIDset)]

    nnodes = gdfsub.shape[0]
    print ('in plotting, number of comids ', nnodes)
    fig,ax=plt.subplots(1,1,figsize=(12,8))
        
    gdf.plot( ax=ax, edgecolor='gray', linewidth=0.5, alpha=0.5, legend="nhd+")
    #gdfsub.plot(ax=ax,edgecolor='#909497', linewidth=1.0, marker='o' )
    #as01032020, switch to light blue
    gdfsub.plot(ax=ax,edgecolor='#5DADE2', linewidth=1.0, marker='o' )
    gdfsub['points'] = gdfsub.apply(lambda x: [y for y in x['geometry'].coords], axis=1)

    #for inverse node to comid dict    
    nodeDict={}
    for key,val in comdIDdict.items():
        nodeDict[val]=key
    for irow in range(adj.shape[0]):        
        indices = np.where(adj[irow,irow:]>0)[0]
        if len(indices)>=5:
            for item in indices:
                startComid = nodeDict[irow]
                endComid = nodeDict[item]
                pt1 = gdfsub[gdfsub[comIDColName]==startComid]['points'].tolist()[0][-1]
                pt2 = gdfsub[gdfsub[comIDColName]==endComid]['points'].tolist()[0][-1]
                x0,y0=pt1[0:2]
                x1,y1=pt2[0:2]
                ax.plot([x0,x1],[y0,y1], linewidth=1.0, linestyle='-.', color='#AF7AC5')                
    if not gagelocshp is None:
        gdfgage = gpd.read_file(gagelocshp)
        for index, row in gdfgage.iterrows():
            if not gageSet is None and row[floIDColName] in gageSet:
                gdfgage.loc[[index], 'geometry'].plot(ax=ax, color='red', markersize=40)
            #as02082021 comment this out to avoid confusion, only label the good gages
            #else:
            #    gdfgage.loc[[index], 'geometry'].plot(ax=ax, color='red',markersize=40)
    
    if not basinboundshp is None:
        #add basin boundary
        boundgdf = gpd.read_file(basinboundshp)
        boundgdf.boundary.plot(ax=ax, alpha=0.8, edgecolor='#626567')

    plt.title(f'Watershed: {prettyname},#Reaches={nnodes}')

    plt.savefig('outputs/adj_rivernet{0}.png'.format(watershedname))
    plt.close()



