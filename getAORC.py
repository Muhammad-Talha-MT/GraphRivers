#author: Alex Sun
#date: 2.13.2022
#purpose: get AORC forcing from AWS
#conv env: waterml
#=======================================================================================
import xarray as xr
import fsspec
import numpy as np
import s3fs
import pandas as pd
from dask.distributed import Client, LocalCluster, progress
import matplotlib.pyplot as plt
import time
import geopandas as gpd

def getNWMDS(nwm_bucket,store_name, datatype='zarr'):
        """
        @see https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem
        """
        config = {'region_name': 'us-east-1'}
        fs = s3fs.S3FileSystem(anon=True, config_kwargs=config)    
        full_path = f"{nwm_bucket}/{store_name}"
        store = s3fs.S3Map(root=full_path, s3=fs)
        
        if datatype=='zarr':
            ds = xr.open_zarr(store, consolidated=True)

        return ds

def getPrecip(ds, region):
    import rioxarray 

    if region=="texas":
        starttime = time.time()
        da = ds['RAINRATE']

        fig,ax = plt.subplots(1,1,figsize=(10,10))
        da = da.sel(time=slice('2017-08-29','2017-08-30'))
        da = da.isel(time=0)
        gdf = gpd.read_file('conus_nwm.shp')
        da.rio.to_raster("nwm.tif")
        with rioxarray.open_rasterio("nwm.tif", masked=False) as xds:
            xds = xds.where(xds>0.0)
            xds.plot(ax=ax)
        gdf.plot(facecolor='none',edgecolor='black', ax=ax, alpha=0.8)
        plt.savefig('testaorcconus.png')
        plt.close()
        print ('time elapsed', time.time()-starttime)

        starttime = time.time()
        da =da.sel(x=slice(-901220.647317,331149.618256), y=slice(-1560010.464445,-358916.842620))
        #debugging
        print (da.shape)
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        print (da.shape)    
        da.where(da>0.0).plot.imshow(ax=ax)
        plt.savefig('testaorctexas.png')
        plt.close()
        print ('time elpased', time.time()-starttime)
        


def main():
    cluster = LocalCluster(n_workers=4)
    # explicitly connect to the cluster we just created
    client = Client(cluster)
    nwm_bucket = "noaa-nwm-retrospective-2-1-zarr-pds"
    ds = getNWMDS(nwm_bucket=nwm_bucket, store_name = "precip.zarr")
    print (ds)
    getPrecip(ds, region='texas')
if __name__ == '__main__':
    main()



