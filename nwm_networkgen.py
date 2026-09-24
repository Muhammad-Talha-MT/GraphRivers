#author: alex sun
#date: 05052021
#date: 05212021, migrate to $WORK2
#date: 12282021, revise formRiverNetwork to return a dictionary
#                add ftype to the dictionary to include reservoir nodes
#                ftype=1, stream, ftype=2, artificial path
#date: 01312022, add NWM2.1 download, controlled by nwm_ver parameter
#date: 09132022, add AORC data getAORC4kmforcing
#====================================================================
from logging import debug
import os
import pickle as pkl
from platform import node
import numpy as np
import sqlite3
import fiona
import pandas as pd
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopandas import GeoSeries

import time,sys
from datetime import datetime

from nwmutils import plotNetwork, plotFPPNetwork, \
    compareNWMUSGS,getNWISData,comparePRMS2USGS,  \
    readShp,readPPShp
from nwmrenyi import NWMSubsetRenci
from nwmaws import NWM21
from lp_util import OrderedSet
from auxilliarydata import getDayMetForcing,getStaticData,getNLDASForcing,getAORC4kmforcing
from basinheatmap import assignNodes2Basin

DEBUG=False

class NHDplus2():
    """Support data manipulatons using NHDPlus2
    sqlite DBs need to be present. Run Buildnhddb.py before this to build the DBs
    dbroot: root directory for the NHDPlus DB
    """
    def __init__(self,dbroot=""):
        #specifiy sqlite db locations
        nhdDBPath = os.path.join(dbroot, "NHDPlusDB.sqlite")
        self.gdb = os.path.join(dbroot, "NHDPlusV21_National_Seamless_Flattened_Lower48.gdb")

        print ('reading NHDPlusDB in ', nhdDBPath)
        if not os.access(nhdDBPath, os.R_OK):
            raise IOError(errno.EACCES, "The database at %s is not readable" %
                      nhdDBPath)        
        catchmentFeatureDBPath = os.path.join(dbroot, "Catchment.sqlite")
        if not os.access(catchmentFeatureDBPath, os.R_OK):
            raise IOError(errno.EACCES, "The catchment feature DB at %s is not readable" %
                      catchmentFeatureDBPath)        
        ogrCmdPath="/work2/02248/alexsund/maverick/anaconda3/envs/tensorflow/bin/ogr2ogr"
        outputDir = "outputs"
        if not os.path.isdir(outputDir):
            raise IOError(errno.ENOTDIR, "Output directory %s is not a directory" % (outputDir,))
        if not os.access(outputDir, os.W_OK):
            raise IOError(errno.EACCES, "Not allowed to write to output directory %s" % (outputDir,))
        outputDir = os.path.abspath(outputDir)
        # Connect to DB
        self.conn = sqlite3.connect(nhdDBPath)
        self.setFlowLineSRC()
  
    def formRiverNetwork(self, fullComIDSet):       
        """Constructing river network from nodes
        as01052022, switch drainage from list to a comid:totDA dictionary to avoid issue
        Parameters
        ----------
        fullComIDset, a set of all nodes to be traversed
        """
        def findRecursiveNeighbor(node0,node1,i):
                if ftype[i] == 0:
                    return node1
                else:
                    for ix,item in enumerate(startNode):
                        if item == node1:
                            return findRecursiveNeighbor(node1,endNode[ix],ix)        
        assert(self.conn)
        nodeDict = {}
        nodeIndxDict={} 
        for nodeIndx,comid in enumerate(fullComIDSet):
            nodeDict[comid] = nodeIndx
            nodeIndxDict[nodeIndx] = comid
        print ('Number of nodes is ', len(fullComIDSet))
        endNode=[]
        startNode=[]
        num_nodes = len(fullComIDSet)
        edgeLen = []
        draingeArea = []
        ftype= []
        for comid in fullComIDSet:
            draingeArea.append(self.getNodeAttribute(self.conn,comid,attr='totDA'))
            #get downstream neighbors
            neighbors = self.getPostdecessors(self.conn, comid)
            if DEBUG: print (f'start={comid}, ',end='')
                
            for ix,item in enumerate(neighbors):
                if item['comID'] in nodeDict.keys():
                    if DEBUG: print (f'end={item["comID"]}, ', end='')
                    #store node index
                    startNode.append(nodeDict[comid])
                    endNode.append(nodeDict[item['comID']])
                    edgeLen.append(item['LENGTHKM'])   
                    if DEBUG: print (f"edge len {item['LENGTHKM']},",end='')

                    if item['FTYPE'] == 'StreamRiver':
                        ftype.append(0)
                    elif item['FTYPE'] == 'ArtificialPath':
                        ftype.append(1)
                    else:
                        ftype.append(2)
            if DEBUG: print (f"FTYPE {ftype}")
        print ('num of edges', len(edgeLen))
        #01/13/2022, remove artificial paths and connect nodes directly
        extraReachesNode=[]
        extraReachesCOMId=[]
        for ireach in range(len(startNode)):
            if ftype[ireach] == 1:
                #search for nodes recursively until a streamriver is reached
                newnode = findRecursiveNeighbor(startNode[ireach],endNode[ireach],ireach) 
                try: 
                    extraReachesCOMId.append([nodeIndxDict[startNode[ireach]], nodeIndxDict[newnode]])
                except KeyError:
                    print ('bad node', newnode )
                extraReachesNode.append([startNode[ireach],newnode])

        print ('extrapaths', extraReachesNode)
        print ('extrapaths', extraReachesCOMId)

        resDict={
            'num_nodes': num_nodes,
            'start_node' : startNode,
            'end_node' : endNode,
            'node_dict': nodeDict,
            'edge_len' : edgeLen,
            'drainage_area': draingeArea,
            'ftype' : ftype,
            'extra_path' : extraReachesCOMId,
            'extra_path_nodes': extraReachesNode,
        }
        return resDict
    
    def getReachesforStreamGauge(self, reachcode, measure, maxdepth=50,minTotalDA=10.0,reGen=True, getAll=False):
        """ generate shp file of reaches for the selected gauge
        Parameters
        ----------
        reachcode, nhdcode of the reach of the gauge, this is available from gageloc.shp e.g., "12090204000060"
        measure, the percentage of the reach, available from gageloc.shp, e.g., measure = 68.71904            
        maxdepth, max depth to traverse upstream
        minTotalDA, minimum drainage area in km2
        reGen, True to regenerate the shapefile
        genAll, True to search for all upstream nodes (very slow, not recommended)        
        """
        assert(self.conn)
        starttime = time.time()
        comID = self.getComIdForStreamGage(self.conn, reachcode, measure)        
        print (comID)

        if getAll:
            allUpstreamReaches=[]
            self.getUpstreamReachesSQL(self.conn, comID, allUpstreamReaches)
        else:
            allUpstreamReaches=self.getFirstOrderUpstreamReaches(self.conn, comID, 
                        maxdepth=maxdepth,minTotalDA=minTotalDA)
            
        #print (allUpstreamReaches)        
        print ('time elapsed ', time.time()-starttime)
        #append the comID itself
        allUpstreamReaches.insert(0, comID)
        self.plotReaches(allUpstreamReaches,reachcode,reGen=reGen)
        pkl.dump(allUpstreamReaches, open('{0}.pkl'.format(reachcode), 'wb'))
        return allUpstreamReaches

        """
        #what's this for?
        comidDictPath='data/comidDict_NHDPlusV21.json'
        self.setFlowLineSRC()
        self.setPlusFlowSRC()        
        if not os.path.isfile(comidDictPath):    
            print ('constructing comidDict ......')
            comidDict = {self.flowlineSRC[f]['properties']['COMID']: f for f in self.flowlineSRC.keys()}            
            with open(comidDictPath, 'w') as fp:
                json.dump(comidDict, fp, indent=2)
        else:
            comidDict = json.load(open(comidDictPath, 'r'))   
        """

    def setPlusFlowSRC(self):
        """Get handle to PlusFlow in GDB (not used)
        """
        self.plusflowSRC =  fiona.open(self.gdb, layer='PlusFlow')

    def setFlowLineSRC(self):
        """Get handle to NHDFlowLine in GDB
        """
        self.flowlineSRC = fiona.open(self.gdb, layer='NHDFlowline_Network')        

    def getComIdForStreamGage(self, conn, reachcode, measure):
        """ Uses NHDFlowline and/or NHDReachCode_ComID table(s) to lookup the ComID associated with a stream gage
            identified by reach code and measure.
        Parameters
        ----------
        conn, An sqlite3 connection to a database that has the NHDFlowline and NHDReachCode_Comid tables
        reachcode, An string representing the Reachcode
        measure, A float representing the measure along reach where Stream Gage is located 
                in percent from downstream end of the one or more NHDFlowline features that are 
                assigned to the ReachCode (see NHDPlusV21 GageLoc table)
        Return
        -------    
        Return An integer representing the ComID associated with a Reachcode, or -1 if a reach with Reachcode 
            was not found.
        """
        comID = -1
        cursor = conn.cursor()
        # NHDPlusV21:
        ## The NHDFlowline comid for a stream flow gage location can be determined from the 
        ## PlusFlowlineVAA where Gage_Loc.Reachcode = PlusFlowlineVAA.Reachcode and 
        ## Gage_Loc.measure => PlusFlowlineVAA.FromMeas and Gage_Loc.measure <= PlusFlowlineVAA.ToMeas.
        cursor.execute("""SELECT p.ComID FROM PlusflowlineVAA as p
        JOIN Gage_Loc as g ON p.ReachCode=g.ReachCode
        WHERE (g.Measure >= p.FromMeas AND g.Measure <= p.ToMeas)
        AND g.ReachCode=? AND g.Measure=?""", (reachcode, measure))
        result = cursor.fetchone()
        if None != result:
            comID = result[0]

        return comID

    def getPredecessors(self, conn, comID):
        """ Get the immediate predecessors of the NHDPlus2 PlusFlow feature of comID
        12/14/2020: get 
            @param conn A connection an SQLite3 database
            @param comdID String representing the ComID of the reach whose immediate predecessor reaches are to be discovered            
            @return A list of immediate predecessor nodes in the NHDPlus2 PlusFlow graph
        """
        immediatePredecessors = []
        cursor = conn.cursor()
        cursor.execute("""
        SELECT p.FROMCOMID,p.TotDASqKM, g.LENGTHKM FROM PlusFlow as p
        JOIN NHDFlowline as g ON p.TOCOMID=g.COMID
        WHERE p.TOCOMID=?""",(comID,))
        for row in cursor:
            immediatePredecessors.append({'comID':row[0],'TotDA':row[1],'LENGTHKM':row[2]})
        return immediatePredecessors

    def getPostdecessors(self, conn, comID):
        """ Get the downstream link of the NHDPlus2 PlusFlow feature of comID
        Updated 06/15/2021: change to postdecessor 
            @param conn, A connection to an SQLite3 database
            @param comdID, String representing the ComID of the reach whose immediate predecessor reaches are to be discovered            
            @return A list of immediate postdecessor nodes in the NHDPlus2 PlusFlow graph
        """
        immediatePostdecessors = []
        cursor = conn.cursor()
        cursor.execute("""
        SELECT p.TOCOMID, g.LENGTHKM, g.FTYPE FROM PlusFlow as p
        JOIN NHDFlowline as g ON p.TOCOMID=g.COMID
        WHERE p.FROMCOMID=?""",(comID,))

        for row in cursor:
            immediatePostdecessors.append({'comID':row[0],'LENGTHKM':row[1],'FTYPE':row[2]})
        return immediatePostdecessors

    def getNodeAttribute(self, conn, comID,attr='totDA'):
        """ get node attribute for given comID
            @param conn, A connection to an SQLite3 database
            @param comdID, String representing the ComID of the reach whose immediate predecessor reaches are to be discovered            
            @param attr, node attribute to query
            
            @return node attribute value
        """
        cursor = conn.cursor()
        if attr == 'totDA':
            cursor.execute("""SELECT TotDASqKM from PlusFlow WHERE FROMCOMID=?""",(comID,))
        
        for row in cursor:
            return float(row[0])
        
    def getPlusFlowPredecessors(self, conn, comID):
        """ Get the immediate predecessors of the NHDPlus2 PlusFlow feature of comID
        
            @param conn A connection an SQLite3 database
            @param comdID String representing the ComID of the reach whose immediate predecessor reaches are to be discovered
            
            @return A list of immediate predecessor nodes in the NHDPlus2 PlusFlow graph
            This is the fiona way, very slow
            filtered = filter(lambda f: f['properties']['TOCOMID']==comID, self.plusflowSRC)
            precessors=[item['properties']['FROMCOMID'] for item in filtered]
            
        """
        immediatePredecessors = []
        cursor = conn.cursor()
        cursor.execute("""SELECT FROMCOMID,TotDASqKM FROM PlusFlow WHERE TOCOMID=?""", (comID,))
        for row in cursor:
            immediatePredecessors.append({'comID':row[0],'TotDA':row[1]})
        return immediatePredecessors

    def getUpstreamReachesSQL(self, conn, comID, allUpstreamReaches):
        """ Recursively searches PlusFlow table in an SQLite database for all stream reaches
            upstream of a given reach.
        
            @note This method has no return value. Upstream reaches discovered are appended to allUpstreamReaches list.
        
            @param conn A connection to an SQLite3 database
            @param comID The ComID of the reach whose upstream reaches are to be discovered
            @param allUpstreamReaches A list containing integers representing comIDs of upstream reaches
        """
        upstream_reaches = self.getPlusFlowPredecessors(conn, comID)
        #print (upstream_reaches)
        if len(upstream_reaches) == 0:
            return
        
        if len(upstream_reaches) == 1 and upstream_reaches[0]['comID'] == 0:
            # We're at a headwater reach
            return

        # For each reach upstream of this reach
        for u in upstream_reaches:
            # Record the upstream reach
            allUpstreamReaches.append(u['comID'])
            # Find reaches upstream of it
            self.getUpstreamReachesSQL(conn, u, allUpstreamReaches)

    def getFirstOrderUpstreamReaches(self, conn, comID, maxdepth=30,minTotalDA=20.0):
        """ Search for first-order upstream reaches in the specified set.
        
            @param config A Python ConfigParser containing the following
            sections and options:
                'NHDPLUS2' and option 'PATH_OF_NHDPLUS2_DB' (absolute path to
                SQLite3 DB of NHDFlow data)
            @param comID The ComID of the reach whose first-order upstream reaches are to be discovered
            @param comIdsInSet A set containing candidate comids
            @param upstreamReaches List containing integers representing comIDs of upstream reaches in set comIdsInSet
            @param maxdepth Integer representing maximum depth of recursion
            
            @return Set containing first order upstream reaches in set
            
            @raise ConfigParser.NoSectionError
            @raise ConfigParser.NoOptionError
        """        
        upstreamReaches = set()
        depth = 0
        self.getFirstOrderUpstreamReachesSQL(conn, comID, upstreamReaches, depth, maxdepth,minTotalDA)
        return list(upstreamReaches)

    def getFirstOrderUpstreamReachesSQL(self, conn, comID, upstreamReaches, depth, maxdepth,minTotalDA):
        """ Recursively search for first-order upstream reaches in the specified set.
        
            @param conn An sqlite3 connection to a database that has NHDPlus2 tables
            @param comID The ComID of the reach whose first-order upstream reaches are to be discovered
            @param comIdsInSet A set containing candidate comids
            @param upstreamReaches Set containing integers representing comIDs of upstream reaches in set comIdsInSet
            @param depth Integer current depth
            @param maxdepth Integer representing maximum depth of recursion
            
        """
        if depth > maxdepth:
            return
        
        upstream_reaches = self.getPlusFlowPredecessors(conn, comID)
        if len(upstream_reaches) == 0:
            return
        
        if len(upstream_reaches) == 1 and upstream_reaches[0]['comID'] == 0:
            # We're at a headwater reach
            return

        # Foreach reach upstream of this reach
        for u in upstream_reaches:
            #print("\timmediate upstream: %s" % (type(u),) )
            # Record the upstream reach if it is in set
            if u['TotDA']>minTotalDA:
                upstreamReaches.add(u['comID'])
            # Keep looking in other upstream branches for first order reaches in set
            self.getFirstOrderUpstreamReachesSQL(conn, u['comID'], upstreamReaches, depth + 1, maxdepth,minTotalDA)
    
    def plotReaches(self, comIDSet, reachcode, reGen=False):
        """This is for testing
        Parameters
        ----------
        comIDSet, set of reach comIDs
        reachcode, the pour point reach node
        reGen: True to generate the shp file
        """
        import cartopy.crs as ccrs
        from cartopy import feature
        from shapely.geometry import shape, MultiLineString
        import matplotlib.pyplot as plt
        if reGen:
            filtered = filter(lambda f: f['properties']['COMID'] in comIDSet, self.flowlineSRC)
            source_crs = self.flowlineSRC.crs
            source_schema = self.flowlineSRC.schema   
            #shp file does not allow datetime, change to str format   
            source_schema['properties']['FDATE']='str:12'
            with fiona.open('testbasin{0}.shp'.format(reachcode),'w',"ESRI Shapefile",
                    crs=source_crs,schema=source_schema) as f:
                for item in filtered: 
                    f.write(item)           
        shp = fiona.open('testbasin{0}.shp'.format(reachcode), 'r')
        mp = MultiLineString([shape(line['geometry']) for line in shp])            
            
        lccProjParams = { 'central_latitude'   : 50.0, # same as lat_0 in proj4 string 
                        'central_longitude'  : -96.0, # same as lon_0
                        'standard_parallels' : (33.0, 45.0) # same as (lat_1, lat_2)
        }
        proj = ccrs.LambertConformal(**lccProjParams)
        shpProj = ccrs.PlateCarree()
        plt.figure()
        ax = plt.axes(projection = proj)
        extent = [-98.783543, -98.673286, 30.654723, 30.935470]
        ax.set_extent([-120.0, -72.0, 22.0, 50.0])
        ax.add_feature(feature.ShapelyFeature(mp, shpProj), edgecolor = 'red')
        
        ax.gridlines()
        plt.title('Rivers with drainage areas greater than 10,000 km$^2$')

        plt.savefig('testnwm.png')
        plt.close()

    def cleanUp(self):
        if self.conn:
            self.conn.close()
        if self.flowlineSRC:
            self.flowlineSRC.close()

def genNWM(watershed, dbroot,shpfilename,reLoad=False, comIDColName='COMID', aggregateRule='24H',nwm_ver='2.0'):    
    """Download nwm data from cuahsi for the 'watershed'
    resample from hourly to daily, and save it as pkl file
    
    Parameters
    ----------
    watershed, name of the watershed
    shpfilename, name of the shp file containing comid
    aggregateRule, for resampling

    Returns
    -------
    comIDset, comid set
    comIDdict,
    nwmdf, dataframe containing 
    goodInd, indices of valid nodes, this is used to construct the adj mat
    gageSet, comid set of all gages
    """
    nwmdf_file = f'nwmdfv20_{aggregateRule}.pkl' if nwm_ver=='2.0' else f'nwmdfv21_{aggregateRule}.pkl'

    if reLoad:
        #get all COMIDs
        comIDset = readShp(shpfilename,comIDColName=comIDColName)
        #get NWM data
        if nwm_ver == '2.0':
            print ('Get NWM2.0 data....')

            nwm = NWMSubsetRenci(comIDset, yearRng=[1993,2018])
            nwm.getData()
            nwmdf = nwm.reSample(nwm.df,rule=aggregateRule) #rule='24H'            
            pkl.dump(nwmdf, open('data/{0}/{1}'.format(watershed,nwmdf_file),'wb'))
            print ('saved nwm dataframe to ', 'data/{0}/{1}'.format(watershed,nwmdf_file))
        else:
            print ("Get NWM2.1 data ")
            print ('number of comids', len(comIDset))
            nwm = NWM21(comIDset)
            nwmdf = nwm.getStreamFlowForBasin(yearRng=[1990,2020], rule=aggregateRule)
            pkl.dump(nwmdf, open('data/{0}/{1}'.format(watershed,nwmdf_file),'wb'))
            print ('saved nwm dataframe to ', 'data/{0}/{1}'.format(watershed,nwmdf_file))

        #[01262022], some comid's may not have data
        badkeys = list(set(comIDset)-set(nwmdf.columns))
        print ('**********Bad keys detected: ', badkeys)
        #[01262022], form network based on valid comidset only
        goodCOMIDset = nwmdf.columns
        print ('Generate river network ....')
        a = NHDplus2(dbroot=dbroot)
        resDict = a.formRiverNetwork(goodCOMIDset)        

        #!!!!warning: these include all comID!!!!
        print ('Saving resdict')
        if nwm_ver=='2.0':
            pkl.dump(resDict, open('data/{0}/adjinfo.pkl'.format(watershed),'wb'))
        else:
            pkl.dump(resDict, open('data/{0}/adjinfo{1}.pkl'.format(watershed,nwm_ver),'wb'))

        """
        plt.figure()
        sns.heatmap(nwmdf)
        plt.savefig('delaware_heatmap.png')
        plt.close()
        """   
    else:
        #note: use comid from the nwmdf header so that column order is right
        nwmdf = pkl.load(open('data/{0}/{1}'.format(watershed,nwmdf_file),'rb'))
        if nwm_ver=='2.0':
            resDict = pkl.load(open('data/{0}/adjinfo.pkl'.format(watershed),'rb'))
        else:
            resDict = pkl.load(open('data/{0}/adjinfo{1}.pkl'.format(watershed,nwm_ver),'rb'))
    return nwmdf,resDict

def genNWMfromPoints(watershed,shpfilename,reLoad=False, comIDColName='COMID', aggregateRule='24H',nwm_ver='2.0', **kwargs):
    """Generate NWM dataset from a set of pour points
    Note: in this case, all the rows in the input shp file are assumed valid
    It's the user's responsibility to form the right subset
    Parameters:
    ----------
    watershed, name of the watershed
    shpfilename, path to the extended flowline shpfile
    reLoad, true to reload nwmdf
    comIDColName, name of the comid column in the shpfile

    Returns:
    ---------
    """
    nwmdf_file = 'nwmdf_fppv20.pkl' if nwm_ver == '2.0' else 'nwmdf_fppv21.pkl'

    if reLoad:
        comIDset = readShp(shpfilename,comIDColName=comIDColName)
        print ('length of comidset', len(comIDset))
        if kwargs['connect_usgsgage']:
            gageFloIDSet = kwargs['gagecomid']
            for item in gageFloIDSet:
                #avoid situations gageloc coincides with pourpoint
                if not item in comIDset:
                    comIDset.append(item)
        print ('length of comidset with extra', len(comIDset))

        print ('Generate river network from point sets (e.g., pour points) ....')

        if nwm_ver=='2.0':
            print (" downloading NWM2.0 data ")
            nwm = NWMSubsetRenci(comIDset, yearRng=[1993,2018])
            nwm.getData()
            nwmdf = nwm.reSample(nwm.df,rule=aggregateRule)#rule='24H'
        else:
            print (" downloading NWM2.1 data ")
            nwm = NWM21(comIDset)
            nwmdf = nwm.getStreamFlowForBasin(yearRng=[1993,2018], rule=aggregateRule)
        #
        if (nwmdf.shape[1]!=len(comIDset)):
            print ('warning: not all comids were retrieved')
        pkl.dump(nwmdf, open('data/{0}/{1}'.format(watershed,nwmdf_file),'wb'))
        print ('saved nwm dataframe to ', 'data/{0}/{1}'.format(watershed,nwmdf_file))
    else:
        #note: use comid from the nwmdf header so that column order is right
        nwmdf = pkl.load(open('data/{0}/{1}'.format(watershed,nwmdf_file),'rb'))
    return nwmdf

def formRiverNetfromPoints(watershed,shpfilename,comIDColName,goodCOMid,**kwargs):
    """Use information in the joined fpp/flowline point shapefile to form network on subbasin pourpoints
    Params
    ------
    goodCOMid: this is passed down from nwm loading
    """
    comIDset,HUCset,downHUCset = readPPShp(shpfilename,comIDColName=comIDColName)
    if kwargs['connect_usgsgage']:
        gageCOMID = kwargs['gagecomid']
        gagehuc12dict = kwargs['gagehuc12dict']
    
    startNode=[]
    endNode = []
    nnode = len(goodCOMid)
    print ('nnode is ', nnode)
    #use order in goodCOMid as node order
    nodeDict = dict(zip(goodCOMid, range(len(goodCOMid))))
    #form lookup table from huc12 to comid
    huc12comidDict={}
    for comid in goodCOMid:
        id = comIDset.index[comIDset==comid]
        if len(id)>0:
            huc12 = HUCset[id.tolist()[0]]        
            huc12comidDict[huc12] = comid

    for index, huc12id in HUCset.items():
        if comIDset[index] in goodCOMid and downHUCset[index] in huc12comidDict.keys():
            #only add edges that are within the watershed
            startNode.append(nodeDict[comIDset[index]])
            endNode.append(nodeDict[huc12comidDict[downHUCset[index]]])
            print (comIDset[index], huc12comidDict[downHUCset[index]])
    
    if kwargs['connect_usgsgage']:       
        for item in gageCOMID:
            #find the huc12 and only add the edge if huc12 is valid
            if gagehuc12dict[item] in huc12comidDict.keys():
                s = nodeDict[item]
                e = nodeDict[huc12comidDict[gagehuc12dict[item]]]
                #exclude the self loop (gage at the pour point already)
                if s!=e:
                    startNode.append(s)
                    endNode.append(e)
                print (item, gagehuc12dict[item],nodeDict[item],nodeDict[huc12comidDict[gagehuc12dict[item]]])
    resDict={
        'start_node':startNode,
        'end_node':endNode,
        'num_nodes':nnode,
    }
    return resDict

def loadFPPNetwork(watershed,plotting=False,connect_usgsgage=False,**kwargs):
    """Load pourpoint based watershsed (HUC12)
    Parameters
    ----------
    connect_usgsgage, if True, add usgs gage as extra comid
    """
    kwargs.setdefault('gageIDset', None)
    kwargs.setdefault('gagelocfile', None) 
    kwargs.setdefault('removebadnode',True)
    kwargs.setdefault('cutoffq', 80) #as 01032022, cutoff percentile
    kwargs.setdefault('boundshp', None)
    kwargs.setdefault('excludeCOMIDs', None)
    kwargs.setdefault('h12shp', None)

    rootdir  = kwargs['rootdir'] 
    shpfilename = kwargs['shpfilename']
    gagelocfile = kwargs['gagelocfile']    
    gageSet = kwargs['gageIDset'] #this is usgs gageid
    gageFloIDSet = kwargs['obsComIDset'] #this is comid
    comIDColName = kwargs['comIDColName']
    removeBadNode =kwargs['removebadnode']
    prettyname = kwargs['prettyname']
    floIDColName = kwargs['floIDColName']
    excludeCOMIDs = kwargs['excludeCOMIDs']
    h12shp = kwargs['h12shp']

    shpfilename = os.path.join(rootdir,shpfilename)
    gagelocfile = os.path.join(rootdir, gagelocfile)
    if not h12shp is None:
        h12shpfile  = os.path.join(rootdir, h12shp)
        
    if kwargs['boundshp'] is not None:
        boundshp = os.path.join(rootdir, kwargs['boundshp'])
    else:
        boundshp = None

    if connect_usgsgage:
        #find huc12 id for each gage location
        assert(not h12shp is None)
        gdfhuc12 = gpd.read_file(h12shpfile)
        gdfgageloc = gpd.read_file(gagelocfile)
        gagehuc12dict={}
        gageCentroids={}
        for item in gageFloIDSet:
            gagegeom= gdfgageloc[gdfgageloc['FLComID']==item].iloc[0].geometry
            gagept = Point(gagegeom.x,gagegeom.y)
            gageCentroids[item] = gagegeom.centroid
            for ix,subbasin in gdfhuc12.iterrows():
                if subbasin.geometry.contains(gagept):
                    gagehuc12dict[item] = subbasin['HUC12']        
        extraparam = {
            'connect_usgsgage': connect_usgsgage,
            'gagecomid': gageFloIDSet,
            'gagehuc12dict':gagehuc12dict,
            'gageCentroids': gageCentroids,
        }
    else:
        extraparam = {
            'connect_usgsgage': False,
        }
    if not os.path.exists(f'data/{watershed}/nwmdf_fpp.pkl'):
        nwmdf = genNWMfromPoints(watershed, shpfilename, reLoad=True, comIDColName=comIDColName,nwm_ver='2.0',**extraparam)
    else:
        nwmdf = pkl.load(open(f'data/{watershed}/nwmdf_fpp.pkl', 'rb'))

    if not os.path.exists(f"data/{watershed}/adjinfo.pkl"):
        print ('saving resdict')
        resDict = formRiverNetfromPoints(watershed,shpfilename,comIDColName,goodCOMid=nwmdf.columns,**extraparam)
        pkl.dump(resDict, open('data/{0}/adjinfo.pkl'.format(watershed),'wb'))
    else:
        resDict = pkl.load(open('data/{0}/adjinfo.pkl'.format(watershed),'rb'))

    comIDdict = dict(zip(nwmdf.columns,range(nwmdf.shape[1])))
    badInd=[]
    for i in range(nwmdf.shape[1]):
        if (nwmdf.iloc[:,i].isna().sum()>0): # or (resDict['ftype'][i] != 0):
            badInd.append(i)    
            print ('bad site', nwmdf.columns[i])
    print (badInd)

    excludeCOMIDs = None
    if not excludeCOMIDs is None:
        for item in excludeCOMIDs:
            badInd.append(comIDdict[item])
    
    goodInd =  list(OrderedSet(range(nwmdf.shape[1]))-OrderedSet(badInd))
    nwmdf = nwmdf.iloc[:, goodInd]
    print ('length of good ind', len(goodInd))
    print ('shape of nwmdf after trimming', nwmdf.shape)
    comIDset = nwmdf.columns.tolist()

    #form dictionary that maps from comid to index in goodInd
    comIDdict=dict(zip(comIDset,range(len(comIDset))))

    if not gageFloIDSet is None:            
        #form a dictionary between usgs comid and usgs gageid
        gageDict={}
        for com_id,usgs_id in zip(gageFloIDSet,gageSet):
            gageDict[com_id] = usgs_id
    else:
        gageDict = pkl.load(open('data/{0}/gageid.pkl'.format(watershed),'rb'))
    print ('gage comid set', gageFloIDSet)

    if plotting:
        print ('Plot network...')
        plotFPPNetwork(watershed, prettyname, shpfilename,comIDset, h12shpfile,
                    gagelocshp=gagelocfile,
                    gageSet=gageFloIDSet, comIDColName=comIDColName,
                    floIDColName=floIDColName,resDict=resDict, basinboundshp=boundshp)

    #generate centroids of reach segments
    print ('in nwm_network', len(comIDset))
    if not os.path.exists('data/{0}/daymetforcing{1}_nwm{2}.pkl'.format(watershed,len(comIDset))):
        print (f'Getting DayMet data for {len(comIDset)} nodes')
        centroids = genReachCentroids(watershed,shpfilename,comIDset,comIDColName,**extraparam)
        assert(len(centroids) == len(comIDset)), 'length of comIDset and centroids do not match'

        pkl.dump(centroids, open('data/{0}/centroids.pkl'.format(watershed),'wb'))
        getDayMetForcing(watershed, comIDset,centroids)
    #load this forcingDF separately in the main code
    #allDF = pkl.load('data/{0}/daymetforcing.pkl'.format(watershed))
    
    #get static data
    """
    if not os.path.exists('data/{0}/staticdata.pkl'.format(watershed)):
        try:
            getStaticData(rootdir,shpfilename,watershed,goodInd,comIDset,comIDColName,addElevSlope=False)
        except Exception:
            print ('Warning: getStaticData was not successful')
            pass
    """
    return comIDset,comIDdict,nwmdf,goodInd,gageDict


def loadNetwork(watershed,plotting=False,compareToUSGS=False,PRMSToUSGS=False,dbroot="",**kwargs):
    """loading river network data

    Parameters
    ---------
    watershed, name of the watershed
    gageSet, subset of observable gage locs
    plotting, True to plot rivernet
    comIDColName: name of the comid column in shapfile
    
    Returns
    -------
    comIDset, subset of good comIDS
    comIDdict,mapper between comID and new column indices in nwmdf
    nwmdf,
    goodInd, subset of good columns relative to the full comID set
            [this is needed to subset the adj matrix]
    gageDict, mapper between comid and usgsid
    Note: 1/26, use cutoffstattype flag to switch median/min cutoff method
    """
    #import matplotlib.pyplot as plt
    kwargs.setdefault('gageIDset', None)
    kwargs.setdefault('gagelocfile', None) 
    kwargs.setdefault('removebadnode',True)
    kwargs.setdefault('compareToUSGS',False)
    kwargs.setdefault('cutoffq', 80) #as 01032022, cutoff percentile
    kwargs.setdefault('boundshp', None)
    kwargs.setdefault('excludeCOMIDs', None)
    kwargs.setdefault('cutoffstattype', 1) #1=median flow, 2=min flow
    kwargs.setdefault('nwm_ver', '2.0') #default nwm version 2.0
    kwargs.setdefault('interval','24H') #set default for NWM output aggregation; if use daymet, must be 1D

    rootdir  = kwargs['rootdir'] 
    shpfilename = kwargs['shpfilename']
    gagelocfile = kwargs['gagelocfile']    
    gageSet = kwargs['gageIDset'] #this is usgs gageid
    gageFloIDSet = kwargs['obsComIDset'] #this is comid
    comIDColName = kwargs['comIDColName']
    removeBadNode =kwargs['removebadnode']
    prettyname = kwargs['prettyname']
    floIDColName = kwargs['floIDColName']
    excludeCOMIDs = kwargs['excludeCOMIDs']
    cutoffstattype = kwargs['cutoffstattype']
    nwm_ver = kwargs['nwm_ver']
    agg_interval = kwargs['interval']
    forcingType = kwargs['forcing_source']
    
    shpfilename = os.path.join(rootdir,shpfilename)
    gagelocfile = os.path.join(rootdir, gagelocfile)
    if kwargs['boundshp'] is not None:
        boundshp = os.path.join(rootdir, kwargs['boundshp'])
    else:
        boundshp = None

    #note: use comid from the nwmdf header so that column order is right
    #note: 04192022, add label for aggregation interval
    nwmdf_file=f'nwmdfv20_{agg_interval}.pkl' if nwm_ver=='2.0' else f'nwmdfv21_{agg_interval}.pkl'
    
    reload=False
    if not os.path.exists(os.path.join(rootdir, nwmdf_file)):
        reload=True
    print ('Loading saved NWM data')
    nwmdf,resDict = genNWM(watershed, dbroot, shpfilename,reLoad=reload,
        aggregateRule = agg_interval,comIDColName=comIDColName,nwm_ver=nwm_ver)

    #detect bad nodes
    #@todo: need to revisit the filtering logic
    #Currently a node is removed if the minimal flow is less than 0.01, or
    #if there is invalid values and the flowpath type is stream (0)
    badInd = []

    if removeBadNode:
        #as01032022, this is tree trimming, find population stat first
        allMIN=[]

        #plot min 
        for i in range(nwmdf.shape[1]):
            if (not nwmdf.iloc[:,i].isna().sum()>0): #and resDict['ftype'][i] == 0:
                if cutoffstattype==1:
                    allMIN.append(np.median(nwmdf.iloc[:,i]))
                elif cutoffstattype==2:
                    allMIN.append(np.min(nwmdf.iloc[:,i]))

        #@backdoor hack 
        #as0125 rethinking the cutoff. A brute-force hard value will create many orphaned nodes
        #If cutoffq>0 it's used as percentile, 
        #If cutoffq<0,its absolute value is used as cutoff
        #percentile
        if kwargs['cutoffq']>0:
            cutoff = np.percentile(allMIN,q=kwargs['cutoffq'])
        else:
            cutoff = abs(kwargs['cutoffq']) # 
            #print ('cutoff', np.percentile(allMIN,q=30)) #for EastTaylor cutt=0.01 at median is about 30 percentile
        print ('cutoff', cutoff)
        if DEBUG:
            plt.figure()
            d = np.array(allMIN)
            plt.hist(d[d>cutoff], bins=10)
            plt.savefig(f'{watershed}_minflow.png')
            plt.close()

        if cutoffstattype==1:
            print (f'----Tree Cutoff Threshold is Median {cutoff} m3/s')
            for i in range(nwmdf.shape[1]):
                if (np.median(nwmdf.iloc[:,i])<cutoff or nwmdf.iloc[:,i].isna().sum()>0): # or (resDict['ftype'][i] != 0):
                    badInd.append(i)

        elif cutoffstattype==2:
            print (f'----Tree Cutoff Threshold is Min {cutoff} m3/s')
            for i in range(nwmdf.shape[1]):
                #as05022022, at hourly level, there'll be na records for houston, so change to 3H
                if (np.min(nwmdf.iloc[:,i])<cutoff or nwmdf.iloc[:,i].isna().sum()>0): # or (resDict['ftype'][i] != 0):
                    badInd.append(i)

        if not excludeCOMIDs is None:
            for item in excludeCOMIDs: 
                badInd.append(nwmdf.columns.get_loc(item))
    print ('length', len(resDict['drainage_area']), nwmdf.shape[1])

    print ('shape of nwmdf before trimming', nwmdf.shape)
    #get good nodes and reset nwmdf    
    goodInd =  list(OrderedSet(range(nwmdf.shape[1]))-OrderedSet(badInd))
    #make column index ascending
    #goodInd = np.sort(np.array(goodInd,dtype=np.long)).tolist()

    nwmdf = nwmdf.iloc[:, goodInd]
    print ('shape of nwmdf after trimming', nwmdf.shape)
    
    comIDset = nwmdf.columns

    """
    #debugging drainage area
    drainageArea = np.array(resDict['drainage_area'])
    for ix, item in enumerate(goodInd):
        print (ix, comIDset[ix], drainageArea[item])
    """
    #form dictionary that maps from comid to index in goodInd
    comIDdict={}
    #comIDset now has the same column order as nwmdf column !!!
    for i,item in enumerate(comIDset):
        comIDdict[item] = i


    if plotting:
        print ('Plot network...')
        plotNetwork(watershed, prettyname, shpfilename,comIDset,gagelocshp=gagelocfile,
                    gageSet=gageFloIDSet, comIDColName=comIDColName,
                    floIDColName=floIDColName,resDict=resDict, basinboundshp=boundshp)

    if compareToUSGS:
        obsComIDset = kwargs['obsComIDset']
        gageIDset = kwargs['gageIDset']

        for gageid,comid in zip(gageIDset,obsComIDset):
            #get usgs data
            #df = getNWISData(gageid=gageid)
            #note: it's better to use cleaned version of usgs data to avoid bad values 
            #compare usgs to nwm data
            compareNWMUSGS(watershed,comID=comid,readData=True, gageID=gageid)
 
    if PRMSToUSGS:
        gageIDset = kwargs['gageIDset']
        gage2prmsDict=kwargs['gage2prmsDict']
        for gageid in gageIDset:
            comparePRMS2USGS(watershed,readData=True, gageID=gageid, gage2prmsDict=gage2prmsDict)

    if not gageFloIDSet is None:            
        #form a dictionary between usgs comid and usgs gageid
        gageDict={}
        for com_id,usgs_id in zip(gageFloIDSet,gageSet):
            gageDict[com_id] = usgs_id
    else:
        gageDict = pkl.load(open('data/{0}/gageid.pkl'.format(watershed),'rb'))
    print ('gage comid set', gageFloIDSet)
    
    #generate centroids of reach segments [daymet only works for daily!!!]
    #assert(agg_interval=='24H')
    assert(forcingType in ['AORC', 'NLDAS', 'Daymet', 'ERA5'])
    if forcingType == 'NLDAS':
        if not os.path.exists('data/{0}/nldasforcing{1}_nwm{2}_{3}.pkl'.format(watershed, len(comIDset), nwm_ver, agg_interval)):
            print ('Getting NLDAS data for NWM', nwm_ver)
            centroids = genReachCentroids(watershed,shpfilename,comIDset,comIDColName)
            pkl.dump(centroids, open('data/{0}/centroids.pkl'.format(watershed),'wb'))
            if nwm_ver == '2.0':
                #getDayMetForcing(watershed, comIDset,centroids, nwm_ver)
                getNLDASForcing(watershed, comIDset,centroids, nwm_ver, agg_interval)
            else:
                #getDayMetForcing(watershed, comIDset,centroids, nwm_ver, startDate='1990-01-01', endDate='2020-12-31', )
                getNLDASForcing(watershed, comIDset,centroids, nwm_ver, agg_interval, startDate='1990-01-01', endDate='2020-12-31')
    elif forcingType == 'AORC':
        if not os.path.exists('data/{0}/aorcforcing{1}_nwm{2}_{3}.pkl'.format(watershed, len(comIDset), nwm_ver, agg_interval)):                              
            print ('Getting AORC data for NWM', nwm_ver)
            centroids = genReachCentroids(watershed,shpfilename,comIDset,comIDColName)
            pkl.dump(centroids, open('data/{0}/centroids.pkl'.format(watershed),'wb'))
            if nwm_ver == '2.0':
                getAORC4kmforcing(watershed, comIDset,centroids, nwm_ver, agg_interval)
            else:
                getAORC4kmforcing(watershed, comIDset,centroids, nwm_ver, agg_interval, startDate='1990-01-01', endDate='2020-12-31')
        #05172023 hack AORC starts on 2000/01/01, so truncate the nwmdf
        nwmdf = nwmdf['2000-01-01': '2021-01-01']

    elif forcingType == 'Daymet':
        if not os.path.exists('data/{0}/daymetforcing{1}_nwm{2}.pkl'.format(watershed, len(comIDset), nwm_ver)):
            print ('Getting Daymet data for NWM', nwm_ver)
            centroids = genReachCentroids(watershed,shpfilename,comIDset,comIDColName)
            pkl.dump(centroids, open('data/{0}/centroids.pkl'.format(watershed),'wb'))
            if nwm_ver == '2.0':
                getDayMetForcing(watershed, comIDset,centroids, nwm_ver)                
            else:
                getDayMetForcing(watershed, comIDset,centroids, nwm_ver, startDate='1990-01-01', endDate='2020-12-31', )
                
    #get static data
    if not os.path.exists('data/{0}/staticdata.pkl'.format(watershed)):
        try:
            getStaticData(rootdir,shpfilename,watershed,goodInd,comIDset,comIDColName,addElevSlope=False,nwm_ver=nwm_ver)
        except Exception:
            print ('Warning: getStaticData was not successful')
            pass

    #as: 0201222, add basin heatmap
    if plotting:
        assignNodes2Basin(watershed=watershed, comIDSubset=comIDset, flowlineshp=shpfilename,
                          comIDColName=comIDColName, nwmdf=nwmdf, nwm_ver=nwm_ver)


    return comIDset,comIDdict,nwmdf,goodInd,gageDict

def genReachCentroids(watershed,flowlineShp,comIDSubset,comIDColName, **kwargs):
    """Generate the centroids of reach (comID) for Daymet download
    returns
    -------
    centriods, mapping between comid and centroid
    """
    gdf = gpd.read_file(flowlineShp)
    gdf = gdf.to_crs('EPSG:4326')
    nodes = gpd.GeoDataFrame(columns=['geometry',comIDColName])
    centroids={}
    for i in range(gdf.shape[0]):
        comID = gdf.iloc[i][comIDColName]
        if comID in comIDSubset:
            centroids[comID] = gdf.iloc[i].geometry.centroid
            nodes = nodes.append({'geometry':Point(centroids[comID]),'comid':gdf.iloc[i][comIDColName]},ignore_index=True)
    #nodes is a shp file, so order does not matter
    kwargs.setdefault('connect_usgsgage', False)
    if kwargs['connect_usgsgage']:        
        for comid,pt in kwargs['gageCentroids'].items():
            if comid in comIDSubset:
                centroids[comid] = pt
    nodes.to_file("data/{0}/{0}_pts.shp".format(watershed))

    return centroids
