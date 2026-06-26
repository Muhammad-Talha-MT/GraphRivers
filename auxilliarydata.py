"""
#author: alex sun
#https://daymet.ornl.gov/web_services

Latitude (required): Enter single geographic point by latitude, value between 52.0N and 14.5N.
Usage Example: lat=43.1

Longitude (required): Enter single geographic point by longitude, value between -131.0W and -53.0W.
Usage Example: lon=-85.3

CommaSeparatedVariables (optional): Daymet parameters include minimum and maximum temperature, precipitation, humidity, shortwave radiation, snow water equivalent, and day length.
Abbreviations:

tmax - maximum temperature
tmin - minimum temperature
srad - shortwave radiation
vp - vapor pressure
swe - snow-water equivalent
prcp - precipitation
dayl - daylength
Usage Example: vars=tmax,tmin
All variables are returned by default.

CommaSeparatedYears (optional): Current Daymet product (version 3) is available from 1980 to the latest full calendar year.
Usage Example: years=2012,2013
Years takes higher precedence over dates.

StartDate & EndDate (optional): Current Daymet product (version 3) is available from 1980 to the latest full calendar year. Date elements follow ISO 8601 convention: YYYY-MM-DD
Usage Example: start=2012-01-31&end=2012-03-31
"""
import sys,os
import pandas as pd
import requests
from datetime import datetime
from datetime import date
import numpy as np
import pickle as pkl
import geopandas as gpd
import xarray as xr
import tqdm 
def getDayMetForcing(watershed,comIDSet, pointSet,nwm_ver, startDate='1993-01-01', endDate='2018-12-31'):
    """Get DayMet forcing data using web service
    #as04192022, added nwm_ver to control different start/end dates
    #fixed a bug syears, which should be the same as the year range shown in startDate/endDate  
    Params
    ------
    comIDSet, subset of ComID used in ML
    pointSet, dictionary of centroid of the ComIDs
    startDate,endDate, defaults to NWM2.0 reanalysis period
    """
    DAYMET_VARIABLES = ['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    requested_vars = ",".join(DAYMET_VARIABLES)
    if nwm_ver == '2.0':
        syears = map(str, range(1993,2019))
    else:
        syears = map(str, range(1990,2021))
    requested_years = ",".join(syears)
    allDF = {}
    for comID in tqdm.tqdm(comIDSet):
        loc = pointSet[comID]
        lat, lon = loc.y, loc.x
        #form web service url
        #url = r"https://daymet.ornl.gov/single-pixel/api/data?lat={0}&lon={1}&vars={2}&start={3}&end={4}&format=json".format(lat,lon,requested_vars,startDate,endDate)
        url = r"https://daymet.ornl.gov/single-pixel/api/data?lat={0}&lon={1}&vars={2}&years={3}&format=json".format(lat,lon,requested_vars,requested_years)

        response = requests.get(url)
        
        if (response.status_code == 200):
            resDict = response.json()['data']
            #dict_keys(['year', 'yday', 'dayl (s)', 'prcp (mm/day)', 'srad (W/m^2)', 'swe (kg/m^2)', 'tmax (deg c)', 'tmin (deg c)', 'vp (Pa)'])
            yarr = resDict['year']
            darr = resDict['yday']

            alldates = [datetime.strptime(f'{int(iyear)}-{int(yday)}', "%Y-%j").strftime("%Y-%m-%d")  
                        for iyear,yday in zip(yarr,darr)]
            alldates = pd.to_datetime(alldates)
            inddates = pd.date_range(startDate,endDate,freq='D')
            df = pd.DataFrame({
                'prcp': resDict['prcp (mm/day)'],
                'srad': resDict['srad (W/m^2)'],
                'swe': resDict['swe (kg/m^2)'],
                'tmax':resDict['tmax (deg c)'],
                'tmin':resDict['tmin (deg c)'],
                'vp': resDict['vp (Pa)'],
            }, index=alldates)

            df = df.reindex(inddates,method='pad')
            allDF[comID] = df
        else:
            print (loc)
            raise Exception('bad request, ', comID)
        pkl.dump(allDF, open('data/{0}/daymetforcing{1}_nwm{2}.pkl'.format(watershed, len(comIDSet), nwm_ver), 'wb'))

def getReachAttributes(watershed,flowlineShp,comIDSubset,comIDColName):
    """Get mean elev and slope for each comid
    Params
    ------
    watershed, watershed name
    flowlineshp, flowline shp file that has elev and slope info in it
                inner join with plusflow and elevslop dbf
    comIDSubset, valid subset of comID
    """
    elevdict={}
    slopedict={}
    streamorderDict={}
    gdf = gpd.read_file(flowlineShp)
    for i in range(gdf.shape[0]):
        comID = gdf.iloc[i][comIDColName]
        if comID in comIDSubset:
            #!!!!NHDPlus elevation in centimeter
            elevdict[comID] = 0.5*(gdf.iloc[i]['MINELEVSMO']+gdf.iloc[i]['MAXELEVSMO'])*0.01
            slopedict[comID]= gdf.iloc[i]['SLOPE']
            streamorderDict[comID] = gdf.iloc[i]['StreamOrde']

    #output in the order of items in comIDSubset
    elev=[]
    slope=[]
    streamorder = []
    for comID in comIDSubset:
        elev.append(elevdict[comID])
        slope.append(slopedict[comID])
        streamorder.append(streamorderDict[comID])
    return np.array(elev),np.array(slope),np.array(streamorder)

def getStaticData(rootDir,flowlineshp,watershed,goodIDSet,comIDSet,comIDColName,addElevSlope=False, nwm_ver='2.0'):
    """Generate static attributes
    @as 1/19, change to use streamorder instead of drainage area
    """
    if nwm_ver=='2.0':
        adjfile = 'data/{0}/adjinfo.pkl'.format(watershed)
    else:
        adjfile = 'data/{0}/adjinfo{1}.pkl'.format(watershed,nwm_ver)
    #note the following info is on the whole node set. It's generated by genrivernet.py
    resDict = pkl.load(open(adjfile, 'rb'))
    draingeArea = resDict['drainage_area']
    draingeArea=np.array(draingeArea)[goodIDSet]    

    #for item in range(len(draingeArea)):
    #    print (item, comIDSet[item], draingeArea[item])

    elev,slope,streamOrder = getReachAttributes(watershed,flowlineshp,comIDSet,comIDColName)
    #remove bad values
    slope[slope<0]=0.0
    #normalize the static features    
    #arr = np.c_[draingeArea, np.array(elev), slope]        

    print ('add elevation and stream order as static attrs')
    #arr = draingeArea.reshape(-1,1)
    arr = np.c_[streamOrder,elev]
        
    print ('static mean', np.nanmean(arr,axis=0), 'std', np.nanstd(arr,axis=0))
    #arr = 2*(arr - np.nanmin(arr,axis=0))/(np.nanmax(arr,axis=0)-np.nanmin(arr,axis=0))-1.0
    #scale to (0,1)
    #arr = (arr - np.nanmin(arr,axis=0))/(np.nanmax(arr,axis=0)-np.nanmin(arr,axis=0))
    arr = (arr - np.nanmean(arr,axis=0))/np.nanstd(arr,axis=0)
    pkl.dump(arr, open('data/{0}/staticdata.pkl'.format(watershed), 'wb'))
    
def getNLDASForcing(watershed,comIDSet, pointSet,nwm_ver, aggregateRule='1H', startDate='1993-01-01', endDate='2018-12-31',ncfile='houston_nldasforcing1993_2021.nc'):
    """as05012022, get nldas data
    """
    ncpath = os.path.join(f'data/{watershed}', ncfile)
    CELLSIZE=0.125 #deg
    dx,dy = (CELLSIZE*0.7,CELLSIZE*0.7)
    allDF = {}

    if not aggregateRule=='1H':
        ncpathagg = os.path.join(f'data/{watershed}', 'nldasforcing_{0}.nc'.format(aggregateRule))
        if not os.path.exists(ncpathagg):
            assert(os.path.exists(ncpath))
            nldasDS = xr.open_dataset(ncpath)
            print ('doing aggregation....')
            #apply different rules on different variables        
            #apcpsfc   (time, lat, lon) float32 ...
            #dlwrfsfc  (time, lat, lon) float32 ...
            #dswrfsfc  (time, lat, lon) float32 ...
            #spfh2m    (time, lat, lon) float32 ...
            #tmp2m     (time, lat, lon) float32 ...
            #ugrd10m   (time, lat, lon) float32 ...
            #vgrd10m   (time, lat, lon) float32 ...
            precip = nldasDS.apcpsfc.resample(time=aggregateRule).sum()
            longwave = nldasDS.dlwrfsfc.resample(time=aggregateRule).mean()
            shortwave = nldasDS.dswrfsfc.resample(time=aggregateRule).mean()
            sh   = nldasDS.spfh2m.resample(time=aggregateRule).mean()
            tmp2m = nldasDS.tmp2m.resample(time=aggregateRule).mean()
            #combine to a single dataset
            nldasDS = xr.Dataset(
                data_vars=dict( 
                    apcpsfc=(["time", "lat", "lon"], precip.values),
                    dlwrfsfc=(["time", "lat", "lon"], longwave.values),
                    dswrfsfc=(["time", "lat", "lon"], shortwave.values),
                    spfh2m = (["time", "lat", "lon"], sh.values),
                    tmp2m = (["time", "lat", "lon"], tmp2m.values),
                ),
                coords=precip.coords,
                attrs=dict(description="nldas data",
                    title = "0.125 Degree Hourly Primary Forcing Data for NLDAS-2",
                    Conventions = "COARDS\nGrADS",
                    dataType = "Grid",
                    history = "Sun May  1 17:23:49 2022: ncks -O -v apcpsfc,dlwrfsfc,dswrf...",
                    NCO = "netCDF Operators version 5.0.6")
            )                
            
            nldasDS.to_netcdf(ncpathagg)
            print (nldasDS)
        else:            
            nldasDS = xr.open_dataset(ncpathagg)
    else:
        assert(os.path.exists(ncpath))
        nldasDS = xr.open_dataset(ncpath)
        #select a subset of variables
        nldasDS = nldasDS[['apcpsfc','dlwrfsfc','dswrfsfc','spfh2m','tmp2m']]

    for comID in comIDSet:
        #extract nldas forcing time series for each comID
        loc = pointSet[comID]
        loc_lat, loc_lon = loc.y, loc.x
        ds = nldasDS.sel(lat=slice(loc_lat-dy,loc_lat+dy), lon=slice(loc_lon-dx,loc_lon+dx), time=slice(startDate,endDate))
        if (len(ds.coords['lat'])>1 or len(ds.coords['lon'])>1):
            ds = ds.isel(lat=0,lon=0)
        elif (len(ds.coords['lat'])==0 or len(ds.coords['lon'])==0):
            raise Exception('bad location')
        
        df = ds.to_dataframe() [['apcpsfc', 'dlwrfsfc', 'dswrfsfc', 'spfh2m', 'tmp2m']]
        allDF[comID] = df

    pkl.dump(allDF, open('data/{0}/nldasforcing{1}_nwm{2}_{3}.pkl'.format(watershed, len(comIDSet), nwm_ver, aggregateRule), 'wb'))


def getAORC4kmforcing(watershed,comIDSet, pointSet, nwm_ver, aggregateRule='1H', startDate='2000-01-01', endDate='2018-12-31'):
    #Note: aorc data is download using downloadAORC.py
    #      aorc data is compiled into zarr by using dataspac.py
    #======================================================================================================================
    repo_dir = '/corral-tacc/utexas/musikal-project/aorc4km/zarr'
    zarrfile = os.path.join(repo_dir, "AORC_4KM_WGRFC.zarr")

    aorcDS = xr.open_zarr(zarrfile)   
    
    dy = aorcDS.latitude[1]-aorcDS.latitude[0] #in degrees
    dx = aorcDS.longitude[1]-aorcDS.longitude[0] #in degrees
    allDF = {}
    for comID in tqdm.tqdm(comIDSet):
        #extract p/t time series for each comID
        loc = pointSet[comID]
        loc_lat, loc_lon = loc.y, loc.x
        ds = aorcDS.sel(latitude=slice(loc_lat-dy,loc_lat+dy), longitude=slice(loc_lon-dx,loc_lon+dx), time=slice(startDate,endDate))
        

        if (len(ds.coords['latitude'])>1 or len(ds.coords['longitude'])>1):
            ds = ds.isel(latitude=0,longitude=0)
        elif (len(ds.coords['latitude'])==0 or len(ds.coords['longitude'])==0):
            raise Exception('bad location')

        df = ds.to_dataframe() [['APCP_surface', 'TMP_2maboveground']]
        df.columns = ['apcpsfc', 'tmp2m']

        if aggregateRule != '1H':
            df1 = df['apcpsfc'].resample(aggregateRule).sum()
            df2 = df['tmp2m'].resample(aggregateRule).mean()
            df = pd.concat([df1,df2], axis='columns')
        allDF[comID] = df
    outfilename = 'data/{0}/aorcforcing{1}_nwm{2}_{3}.pkl'.format(watershed, len(comIDSet), nwm_ver, aggregateRule)
    print ('dump to ', outfilename)
    pkl.dump(allDF, open(outfilename, 'wb'))

def test():    
    pointSet=[{'x':-76.9481570913146, 'y':41.23743272518211}]
    getDayMetForcing(pointSet)
if __name__ == '__main__':
    test()