#author: alex sun
#date: 12282021
#purpose: retrieve daily data from aws s3 bucket
#https://github.com/lsterzinger/fsspec-reference-maker-tutorial/blob/main/tutorial.ipynb
#updated: 1/20/2022, converted to NWM21 class
#=================================================================================================
import xarray as xr
import fsspec
import numpy as np
import s3fs
import pandas as pd
from dask.distributed import Client, LocalCluster, progress
import matplotlib.pyplot as plt
import time
import sys

class NWM21():
    def __init__(self, comIDset, store_name="chrtout.zarr") -> None:
        """
        Params
        ------
        comIDset, a set of COMIDs
        store_name, name of the zarr store
        """
        #is this the best place to start cluster
        cluster = LocalCluster(n_workers=4)
        # explicitly connect to the cluster we just created
        client = Client(cluster)

        self.comIDset = comIDset
        nwm_bucket = "noaa-nwm-retrospective-2-1-zarr-pds"
        print ('getting s3 store')
        self.ds = self.getNWMDS(nwm_bucket, store_name)
        
    def getNWMDS(self, nwm_bucket,store_name, datatype='zarr'):
        """
        @see https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem
        """
        config = {'region_name': 'us-east-1'}
        fs = s3fs.S3FileSystem(anon=True, config_kwargs=config)    
        full_path = f"{nwm_bucket}/{store_name}"
        store = s3fs.S3Map(root=full_path, s3=fs)
        
        if datatype=='zarr':
            ds = xr.open_zarr(store, consolidated=True)
        else:
            raise NotImplementedError('data type not supported')
        """
            #this is the content of the ds printout for chrout
            <xarray.Dataset>
            Dimensions:     (feature_id: 2776738, time: 367439)
            Coordinates:
                elevation   (feature_id) float32 dask.array<chunksize=(2776738,), meta=np.ndarray>
            * feature_id  (feature_id) int32 101 179 181 ... 1180001803 1180001804
                gage_id     (feature_id) |S15 dask.array<chunksize=(2776738,), meta=np.ndarray>
                latitude    (feature_id) float32 dask.array<chunksize=(2776738,), meta=np.ndarray>
                longitude   (feature_id) float32 dask.array<chunksize=(2776738,), meta=np.ndarray>
                order       (feature_id) int32 dask.array<chunksize=(2776738,), meta=np.ndarray>
            * time        (time) datetime64[ns] 1979-02-01T01:00:00 ... 2020-12-31T23:0...
            Data variables:
                crs         |S1 ...
                streamflow  (time, feature_id) float64 dask.array<chunksize=(672, 30000), meta=np.ndarray>
                velocity    (time, feature_id) float64 dask.array<chunksize=(672, 30000), meta=np.ndarray>
            Attributes:
                TITLE:                OUTPUT FROM WRF-Hydro v5.2.0-beta2
                code_version:         v5.2.0-beta2
                featureType:          timeSeries
                model_configuration:  retrospective
                proj4:                +proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1...    
        """
        return ds

    def getStreamFlowForBasin(self, yearRng: tuple, resample=False, rule:str='1D') -> pd.DataFrame:
        print ('getting s3 streamflow')
        time_slice = slice(f'{yearRng[0]}/01/01', f'{yearRng[1]}/12/31')
        #get streamflow        
        starttime= time.time()
        #get list of comids in ds
        comlist = self.ds['streamflow']['feature_id'].values.tolist()
        validCOMID = list(set(comlist).intersection(set(self.comIDset)))
        print ('Number of valid COMIDs', len(validCOMID))
        
        daQ = self.ds['streamflow'].sel(feature_id=validCOMID)
        daQ = daQ.sel(time=time_slice)
        if resample:
            daQ = daQ.resample(time=rule).mean()
        arr = daQ.values
        print ('subsetting time', time.time()-starttime)
        
        starttime = time.time()
        bigdf = pd.DataFrame(arr, columns=daQ.coords['feature_id'], index=pd.to_datetime(daQ.coords['time']))
        print ('df generation time', time.time()-starttime)        
        return bigdf

def main():
    #===========testing=======================
    comIDset = [1333198,1333022,1333490,1333564,1333418] #, 1333198,1333022,1333490,1333564,1333418]
    nwm21 = NWM21(comIDset)
    df = nwm21.getStreamFlowForBasin(yearRng=(1993, 2018))
    print (df.columns)
    print (df.shape)
    print (df.head(10))

if __name__ == '__main__':
    main()
