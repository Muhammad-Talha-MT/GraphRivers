#author: alex sun
#desc: generate rivernetwork
#date: 02/02/2021
#rev: 04262021, clean up for github
#rev: 05052021, separate network generation from data
#rev: 06022021, add custom floIDColName
#rev: 03212022, customize for musikal
#rev: 04182022, focus on houston watershed
#rev: 04232023, cleanup for sharing
#=====================================================================
from nwm_networkgen import loadNetwork,loadFPPNetwork
from nwmutils import checkUSGSGauge
from util_gtnet import load_config
import os

class Watershed():
    """Parent class of Watershed
    """
    def __init__(self, watershed, rootdir, shpfilename, gagelocfile):
        """Init
        
        Parameters
        ----------
        watershed [str]: name of the watershed
        rootdir [str]: directory where all data for the watershed is stored
        shpfilename: name of the shape layer
        gagelocfile: name of the gauge location file
        """

        self.watershed=watershed
        self.rootdir = rootdir
        self.shpfilename = shpfilename
        self.gagelocfile = gagelocfile
        
        self.comIDColName = 'COMID' #column name of comid
        self.removebadnode=True
        self.prettyName = watershed
        self.transformFlow=True
        self.floIDColName='FLComID'
        self.cutoffq = 80 #quantile for trimming the rivernetwork [as: 01/03/2022]

        self.gageIDset = None
        self.obsComIDset = None
        self.gage2prmsDict = None
        self.shpbasinbound = None #for adding bound to rivernet plots [as: 1/10/2022]
        self.excludeCOMIDs = None #for excluding bad comids [as: 1/19/2022]
        self.shptype = 'normal' #for telling whether a basin is fpp basin
        self.h12shp  = None   #for plotting h12 basins in plotfppnetwork
        self.fppshp = None  #name of the fpp shapefile
        self.fppreplacement = None
        
    def __repr__(self):
        return f'watershed:{self.watershed}, rootdir:{self.rootdir}, gageloc:{self.gagelocfile}'

class EastTaylor(Watershed):
    def __init__(self, watershed, rootdir, shpfilename, gagelocfile,basinboundfile=None):
        super(EastTaylor, self).__init__(watershed, rootdir, shpfilename, gagelocfile)
        self.comIDColName = 'ComID'
        self.removebadnode=True
        self.prettyName = 'East Taylor'
        self.shpbasinbound = basinboundfile
        self.gageIDset = ['09112200','09107000','09109000','09110000','09112500'] #usgs gageid
        self.obsComIDset = [1333198,1333022,1333490,1333564,1333418] #FLComID in gageloc.shp
        #this information is from the exashed_segment file [column Exasegment]
        self.gage2prmsDict = {
        '09112200':26,
        '09107000':36, 
        '09109000':42, 
        '09110000':50,
        '09112500':32}
        self.cutoffq = -0.01 #cutoff 0.01 m3/s  (295 node case)
        #self.cutoffq = -0.005 #cutoff 0.01 m3/s  (397 node case)
        #self.cutoffq = 30 #227 node
        #for bad fpp replacement (for easttaylor, h12 pour point may land on the reservoir
        # replace those with the nearest reach comid)
        self.fppreplacement={1333580:1333022}

class Gunnison(Watershed):
    def __init__(self, watershed, rootdir, shpfilename, gagelocfile,basinboundfile=None):
        super(Gunnison, self).__init__(watershed, rootdir, shpfilename, gagelocfile)
        self.comIDColName = 'ComID'
        self.removebadnode=True
        self.prettyName = 'Gunnison'
        self.cutoffq = -0.01 #cutoff 0.01 m3/s, this gives 766 node
        #self.cutoffq = 80 #cutoff 0.01 m3/s
        self.shpbasinbound = basinboundfile
        #as 1/25/
        #  remove 3252173 09128000 
        # and 3232789 (comid), 09144250 (usgsid)

        self.gageIDset = ['09147025', '09124500', '09119000', '09126000', '09113980', '09132500', '09147500', '09112500', 
        '09109000', '09146200', '09107000', '09152500', '09114500', '09112200',  
        '09149500', '09147000', '09110000', '09115500', '09118450'] #usgs gageid        
        self.obsComIDset = [9769294, 3253975, 9773129, 3253985, 3252623, 1337236, 9769790, 1333418, 
        1333490, 9769826, 1333022, 3232569, 3252637, 1333198, 
        9767880, 9769828, 1333564, 9773517, 9773615]       #FLComID in gageloc.shp 

        #these are easttaylor nodes
        #self.gageIDset = ['09112200','09107000','09109000','09110000','09112500'] #usgs gageid
        #self.obsComIDset = [1333198,1333022,1333490,1333564,1333418] #FLComID in gageloc.shp
        
        self.excludeCOMIDs = [9769076, 9769048, 9768176, 9768148, 
        3251401,3251421,3251439,3252185, 3252189,3252263,3252139,3251515,3251629,3251633,3252195] 
        
class CentralTXCoastal(Watershed):
    def __init__(self, watershed, rootdir, shpfilename, gagelocfile,basinboundfile=None):
        super(CentralTXCoastal, self).__init__(watershed, rootdir, shpfilename, gagelocfile)
        self.comIDColName = 'COMID'
        self.removebadnode=True
        self.prettyName = 'Central Texas Coastal'
        #self.cutoffq = -0.01 #cutoff 0.01 m3/s, this gives 766 node
        self.cutoffq = 80 #cutoff 0.01 m3/s
        self.shpbasinbound = basinboundfile

        self.gageIDset = ['08181500', '08167500', '08189700', '08171000', '08164390', '08164300', '08186000', '08164504', '08173900', 
        '08164600', '08164503', '08178800', '08188500', '08171300', '08167000', '08175000', '08166000', '08178880', '08180700', '08165300', 
        '08168500', '08177500', '08162600', '08165500', '08166140', '08172400', '08189200', '08166200', '08169000', '08164000', '08183500', 
        '08172000', '08189500', '08175800', '08167800', '08181800', '08181480', '08176500', '08164800', '08170500'] #usgs gageid        


        self.obsComIDset = [10836420, 3589120, 5297631, 1630223, 7846429, 7846049, 7852265, 7846399, 1622713, 9349285, 7846391, 10840488, 3840125, 
        1631023, 3589508, 1623207, 3585554, 10833740, 10835974, 3585678, 1620031, 1638907, 9355362, 3586192, 3585626, 1631087, 3159657, 3585724, 1619649, 
        7842827, 3838221, 1631387, 5289461, 1637447, 1619595, 10840572, 10836388, 1639225, 9349455, 1631099]       #FLComID in gageloc.shp 

class Houston(Watershed):
    def __init__(self, watershed, rootdir, shpfilename, gagelocfile,basinboundfile=None):
        super(Houston, self).__init__(watershed, rootdir, shpfilename, gagelocfile)
        self.comIDColName = 'COMID'
        self.removebadnode=True
        self.prettyName = 'Houston'
        self.cutoffq = -0.001 #cutoff 0.01 m3/s, this gives 766 node
        #self.cutoffq = 80 #cutoff 
        self.shpbasinbound = basinboundfile
        #self.gageIDset =  ['08075000', '08075400', '08075730', '08075770', '08072730', '08073500', '08073600', '08076000'] #usgs gageid        
        #self.obsComIDset = [1440385, 1439761, 1440389, 1440277, 1440237, 1439317, 1439357, 1440183]    #FLComID in gageloc.shp 
        self.gageIDset =  ['08075000', '08075730', '08075770', '08072730', '08073500', '08073600', '08076000'] #usgs gageid        
        self.obsComIDset = [1440385,   1440389, 1440277, 1440237, 1439317, 1439357, 1440183]    #FLComID in gageloc.shp 
        
class LowerSabine(Watershed):
    def __init__(self, watershed, rootdir, shpfilename, gagelocfile,basinboundfile=None):
        super(LowerSabine, self).__init__(watershed, rootdir, shpfilename, gagelocfile)
        self.comIDColName = 'COMID'
        self.removebadnode=True
        self.prettyName = 'Lower Sabine'
        self.cutoffq = -0.001 #cutoff 0.01 m3/s
        #self.cutoffq = 80 #cutoff 
        self.shpbasinbound = basinboundfile
        self.gageIDset =  ['08026000', '08028000', '08029500', '08025360', '08030500'] #usgs gageid        
        self.obsComIDset =[ 8329788, 8330660, 8330722, 8328658, 8331804]    #FLComID in gageloc.shp 
        
def genRiverNet(watershedObj,dbroot="",compareToUSGS=False, PRMSToUSGS=False, connect_usgsgage=False, 
                cutoffstattype=1, nwm_ver='2.0', interval='24H', forcing_source='aorc'):
    """main driver for generating river net
    Parameters:
    ----------
    watershed: watershed class
    """
    watershed = watershedObj.watershed

    kwargs={
        'rootdir':watershedObj.rootdir,
        'shpfilename':watershedObj.shpfilename,
        'gagelocfile':watershedObj.gagelocfile,
        'gageIDset': watershedObj.gageIDset,
        'obsComIDset':watershedObj.obsComIDset,
        'comIDColName':watershedObj.comIDColName,
        'removebadnode':watershedObj.removebadnode,
        'prettyname': watershedObj.prettyName,
        'gage2prmsDict':watershedObj.gage2prmsDict,
        'floIDColName': watershedObj.floIDColName,
        'cutoffq': watershedObj.cutoffq,
        'boundshp': watershedObj.shpbasinbound,
        'excludeCOMIDs': watershedObj.excludeCOMIDs,
        'h12shp': watershedObj.h12shp,
        'cutoffstattype':cutoffstattype,
        'nwm_ver':nwm_ver,
        'interval':interval,
        'forcing_source': forcing_source
    }    
    #return comIDset,comIDdict,nwmdf,goodInd
    if '_fpp' in watershed or watershedObj.shptype=='fpp':
        return loadFPPNetwork(watershed, plotting=True, connect_usgsgage=connect_usgsgage, **kwargs)
    else:
        return loadNetwork(watershed,
                plotting=False, #!!! uncomment to plot Figure heatmap in the paper 
                compareToUSGS=compareToUSGS, 
                PRMSToUSGS=PRMSToUSGS,
                dbroot=dbroot, **kwargs
            )

def loadWatershed(args, reGen=False, returnWobj=False):
    """
    Parameters
    ---------
    args, container of input arguments
    reGen, True to regenerate the .pkl 
    """

    #dbroot, root directory of NHDPlus SQLite
    #basedir, base directory of code base

    basedir = args.rootdir
    dbroot = args.nhdplusdbroot
    watershed = args.watershed_name

    if watershed=='easttaylor':
        rootdir = '/'.join([basedir, 'data/easttaylor'])
        #flowlinefile = "EastTaylorFlowLine.shp"
        #flowlinefile = "EastTaylorFlowLineElevSlope.shp"
        #as01192022, replace flowline
        flowlinefile = "EastTaylorFlowLine_vaa.shp"
        gagelocfile = "EastTaylorGageLoc.shp"    
        basinboundfile = "EastTaylorProj.shp"
        #checkUSGSGauge(os.path.join(rootdir, gagelocfile))
        wobj = EastTaylor('easttaylor',rootdir,flowlinefile,gagelocfile,basinboundfile=basinboundfile)
        #[0128], for testing imputation capability
        wobj.fppshp = os.path.join(rootdir, 'easttaylor_fpp_joined.shp')
    elif watershed=='gunnison':
        #as02212022
        rootdir = '/'.join([basedir, 'data/gunnison'])
        flowlinefile = "gunnison_flowlines_vaa.shp"
        gagelocfile = "gunnison_gagelocs.shp"
        basinboundfile = "gunnison_bound.shp"    
        #checkUSGSGauge(os.path.join(rootdir, gagelocfile))
        wobj = Gunnison('gunnison',rootdir,flowlinefile,gagelocfile, basinboundfile=basinboundfile)         
        #[0221], add imputation capability
        wobj.fppshp = os.path.join(rootdir, 'gunnison_fpp_joined_proj.shp')
    elif watershed == 'centraltxcoast':
        rootdir = '/'.join([basedir, 'data/centraltxcoast'])
        flowlinefile = "central_texas_flowline_vaa.shp"
        gagelocfile = "central_texas_gageloc.shp"    
        #use this go find good gages
        #checkUSGSGauge(os.path.join(rootdir, gagelocfile))
        #comment the following out when using checkUSGSGauge in the above
        wobj = CentralTXCoastal('centraltxcoastal',rootdir,flowlinefile,gagelocfile)  
    elif watershed == 'houston':
        rootdir = '/'.join([basedir, 'data/houston'])
        flowlinefile = "houston_flowline_vaa.shp"
        gagelocfile = "houston_gage_loc.shp"    
        basinboundfile = "houstonproj.shp"
        #use this go find good gages
        #checkUSGSGauge(os.path.join(rootdir, gagelocfile))
        #comment the following out when using checkUSGSGauge in the above
        wobj = Houston('houston',rootdir,flowlinefile,gagelocfile,basinboundfile=basinboundfile)  
    elif watershed == 'lowersabine':
        rootdir = '/'.join([basedir, 'data/lowersabine'])
        flowlinefile = "lowersabine_flowlinevaa.shp"
        gagelocfile = "lowersabine_gageloc.shp"    
        basinboundfile = "lowersabinebound.shp"
        #use this go find good gages
        #checkUSGSGauge(os.path.join(rootdir, gagelocfile))
        #comment the following out when using checkUSGSGauge in the above
        wobj = LowerSabine('lowersabine',rootdir,flowlinefile,gagelocfile,basinboundfile=basinboundfile)  
    else:
        raise NotImplementedError
    if returnWobj:
        return wobj
    else:
        if wobj.shptype == 'normal':
            return genRiverNet(watershedObj=wobj,dbroot=dbroot,
                compareToUSGS=args.network.compare_to_usgs, 
                PRMSToUSGS=args.network.compare_to_prms,
                cutoffstattype=args.network.cutoffstattype,
                nwm_ver=args.data.nwm_ver, 
                interval=args.data.interval, 
                forcing_source = args.data.forcing_source)
                
        elif wobj.shptype == 'fpp':
            return genRiverNet(watershedObj=wobj,dbroot=dbroot,
                compareToUSGS=args.network.compare_to_usgs, 
                PRMSToUSGS=args.network.compare_to_prms,
                connect_usgsgage=args.network.connect_usgsgage,
                cutoffstattype=args.network.cutoffstattype,
                nwm_ver=args.data.nwm_ver, 
                interval=args.data.interval,
                forcing_source = args.data.forcing_source)

def get_shared_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse arguments for genrivernet",
    )
    parser.add_argument('--watershed_name', type=str, required=True, help='specify watershed name')
    parser.add_argument('--compare_to_usgs', action='store_true', default=False, help='compare to usgs record')
    parser.add_argument('--compare_to_prms', action='store_true', default=False, help='compare to prms results')
    parser.add_argument('--connect_usgsgage', action='store_true', default=False, help='true to attach usgs gage to comid set for fpp watersheds')
    parser.add_argument('--cutoffstattype', type=int, default=1, help='network cutoff stat')
    parser.add_argument('--nwm_ver', type=str, required=True, default='2.0', help='specify watershed name')
    parser.add_argument('--interval', type=str, required=True, default='1H', help='aggregation interval')
    return parser

def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse arguments for genrivernet",
    )
    parser.add_argument('--config', type=str, required=True, help='config file name')
    cargs = parser.parse_args()
    args = load_config(cargs.config)

    loadWatershed(args)

if __name__ == '__main__':
    main()
