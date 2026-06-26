from netCDF4 import Dataset
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class NWMSubsetRenci():
    """Download Hydroshare reanalysis data
    Currently there are two versions of the dataset: 
    A 25-year (January 1993 through December 2017) retrospective simulation using 
    version 1.2 of the National Water Model, and a 26-year (January 1993
     through December 2018) retrospective simulation using version 2.0 of 
     the National Water Model.    

    """
    def __init__(self,comIDSet,yearRng=None):
        self.comIDSet = comIDSet
        if yearRng is None:
            yearRng=[1993,2018]
        self.yearRng = yearRng
    
    def getData(self): 
        pathname  = "http://thredds.hydroshare.org/thredds/dodsC/nwm/retrospective/nwm_v2_retro_full.ncml"
        filehandle = Dataset('[FillMismatch]'+pathname,'r',format="NETCDF4")
        """
        time Hours since 1970-01-01 00:00:00 UTC
        - Spans 1993-01-01 to 2018-12-31
        - Contains 227,903 hourly time steps for
        [time = 0..227903]
        """
        times=filehandle.variables['time'][:]
        year_start,year_end = self.yearRng
        month_start,month_end = [1,12]
        tStart = datetime(year_start,month_start,1,0,0,0)
        tEnd = datetime(year_end,month_end,31,23,59,59)
        time_min = ((tStart-datetime(1970,1,1,0,0,0)).total_seconds())//3600
        time_max = ((tEnd-datetime(1970,1,1,0,0,0)).total_seconds())//3600
        time_index_min = (np.abs(times-time_min)).argmin()
        time_index_max = (np.abs(times-time_max)).argmin()
        time_index_range = range(time_index_min, time_index_max+1)
        #============================
        # Data Subsetting 
        #============================
        #feature_id is the same as ComID
        featureIDs = filehandle.variables['feature_id'][:]
        featureIndex = np.where(np.in1d(featureIDs, self.comIDSet))[0]
        
        datahandle = filehandle.variables['streamflow']
        data = datahandle[featureIndex, time_index_range]
        collabels = []
        for i in featureIndex:
            collabels.append(featureIDs[i])

        self.df = pd.DataFrame(data.T, columns=collabels, index=pd.date_range(start=tStart,end=tEnd,freq='1H'))
        
    def reSample(self, df, rule='24H'):
        """
        rule, the interval for downsampling nwm data
        """        
        #aggregate
        df = df.resample(rule).mean()   
        return df

    def plotStreamFlow(self,df, rule='24H'):
        df = self.reSample(rule)
        cols = df.columns 
        fig,ax=plt.subplots(df.shape[1],1,figsize=(6,16))
        for i in range(df.shape[1]):
            ax[i].plot(df.iloc[:,i].to_numpy(dtype=np.float64))
            ax[i].text(0.01, 0.9, cols[i], fontsize=12, transform=ax[i].transAxes)
        plt.tight_layout(h_pad=0.02)
        plt.savefig('testnwmchrtout.png')
        plt.close()


