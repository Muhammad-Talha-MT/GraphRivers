#author: alex sun
#date: 07182022
#purpose: download nwm data at the intersection between nhdplus network and delft3d boundary
#
import geopandas as gpd
import os
from nwmaws import NWM21
import time
def getCOMIDSet(downloadOption='domain', ftype=None):
    """
    downloadOption, can be domain or boundary
    """
    assert(downloadOption in ['domain', 'boundary'])
    if downloadOption == 'boundary':
        shpfile = 'maps/delftintersect_vaa.shp'
    else:
        shpfile = 'maps/d3dflowlines_all_vaa.shp'
    if not os.path.exists(shpfile):
        raise Warning("shp file does not exist")
    gdf = gpd.read_file(shpfile)

    if not ftype is None:
        #do filtering if necessary
        if type(ftype) is list:
            gdf = gdf.loc[gdf['FTYPE'] in ftype]
        else:
            gdf = gdf.loc[gdf['FTYPE'] == ftype]
    
    print ('number of reaches extracted', gdf.shape[0])
    
    return gdf['COMID'].tolist()

def getNWM21():
    downloadOption = 'domain'
    comIDset = getCOMIDSet(downloadOption=downloadOption)
    starttime = time.time()
    nwm21 = NWM21(comIDset)    
    df = nwm21.getStreamFlowForBasin(yearRng=(2017, 2018))
    print (df.columns)
    print (df.shape)
    print (df.head(10))
    #save the df
    if not os.path.exists('data'):
        os.path.mkdir('data')    
    df.to_csv(f"data/gom_nwm21_{downloadOption}.csv")
    print ('Time taken to download ', time.time()-starttime)

if __name__ == '__main__':
    getNWM21()
