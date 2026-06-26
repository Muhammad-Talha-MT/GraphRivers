#author: Alex Sun
#date: 02012022
#purpose: print subbasin heatmap to show that subbasin nodes are highly correlated with a block structure
#rev date: 02/23/2022, clean up for the paper
#conda env: waterml
#need to call this function assignNodes2Basin from nwm_networkgen.py
#03242022, add network analysis
import os
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import networkx as netx
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def assignNodes2Basin(watershed,comIDSubset,flowlineshp,h12shp='EastTaylor_h12.shp', 
        comIDColName='ComID',nwmdf=None, nwm_ver='2.0'):        
    """Assign each comid to the huc12 basin
       plots basinheatmap when nwmdf is not None
       performs network analysis when adjMat is not None
    """
    
    rootdir = 'data/{0}'.format(watershed)
    gdfcomid = gpd.read_file(flowlineshp)
    #@todo:move this kwargs
    gdfh12 = gpd.read_file(os.path.join(rootdir, h12shp))
    gdfcomid = gdfcomid.to_crs(gdfh12.crs)
    #create a list of centroids for comid reaches
    xv=[]
    yv=[]
    comidArr=[]
    for i in range(gdfcomid.shape[0]):
        comID = gdfcomid.iloc[i][comIDColName]
        if comID in comIDSubset:
            centroid = gdfcomid.iloc[i].geometry.centroid
            xv.append(centroid.x)
            yv.append(centroid.y)            
            comidArr.append(comID)
    s = gpd.GeoSeries(map(Point, zip(xv, yv)))
    #make an empty dict
    basinNodes={}
    for i in range(gdfh12.shape[0]):
        basinNodes[gdfh12.iloc[i]['HUC12']] =[]
    
    #loop through flowline shpfile to assign flowlines to subbasins
    #note: 0201, I manually validated the mapping    
    for j in range(gdfh12.shape[0]):
        for i in range(len(s)):     
            if gdfh12.iloc[j].geometry.contains(s[i]):
                basinNodes[gdfh12.iloc[j]['HUC12']].append(comidArr[i])
    
    
    #now plot heatmap
    if not nwmdf is None:
        outputdir = 'outputs/{0}'.format(watershed)
        #get subset
        sortedcomid = [x for v in basinNodes.values() for x in v]
        #get count of each basin
        nodecount = [len(v) for v in basinNodes.values()]
        df = nwmdf.loc[:,sortedcomid]
        corr = df.corr()

        # plot subbasin on a separate plot?
        fig,ax=plt.subplots(1,1, dpi=300)
        gdfh12.plot(edgecolor='gray', cmap='tab20', alpha=0.8, ax=ax)
        for idx, row in gdfh12.iterrows():
            print (row.geometry.centroid.coords[0])
            ax.annotate(text=idx+1, xy=row.geometry.centroid.coords[0], horizontalalignment='center')        
        plt.savefig(os.path.join(outputdir, 'subbasin_plot.eps'))
        plt.close()

        # plot the heatmap
        fig,ax= plt.subplots(1,1, figsize=(10,10), dpi=300)
        sns.heatmap(corr, cmap='viridis', ax=ax, yticklabels=False, xticklabels=False)
        #width and height are the same
        w = ax.get_xticks()
        
        counter=0
        for ix,item in enumerate(nodecount):
            #ax[1].text(counter+0.5*item, -w[0], ix+1, fontsize=11) #, transform=ax[1].transAxes)
            counter+=item
            ax.hlines(counter-0.5, 0.5, df.shape[0]-0.5, linestyle='-', linewidth=1, color="white")
            ax.vlines(counter-0.5, 0.5, df.shape[0]-0.5, linestyle='-', linewidth=1, color="white")        

        plt.savefig(os.path.join(outputdir, 'nwm_{0}_heatmap.eps'.format(nwm_ver)))
        plt.close()
    

def pltNetworkgraph(args, adjMat,comIDdict, graphtype='full',fpplist=None, comIDColName='ComID', nwm_ver='2.0',
    flowlineshp='EastTaylorFlowLine_vaa.shp',h12shp='EastTaylor_h12.shp',basinboundshp='EastTaylorProj.shp'):        

    """Assign each comid to the huc12 basin
       plots basinheatmap when nwmdf is not None
       performs network analysis when adjMat is not None
       For now, flowlineshp and h12shp are provided without full path
    """
    np.fill_diagonal(adjMat,0.0)
    #output folder
    os.makedirs('outputs', exist_ok=True)
    outputdir = 'outputs/{0}'.format(args.watershed_name)
    os.makedirs(outputdir, exist_ok=True)
    rootdir = 'data/{0}'.format(args.watershed_name)

    gdfcomid = gpd.read_file(os.path.join(rootdir,flowlineshp))
    gdfh12 = gpd.read_file(os.path.join(rootdir, h12shp))
    gdfcomid = gdfcomid.to_crs(gdfh12.crs)

    #create a list of centroids for comid reaches
    xv={}
    yv={}
    if graphtype=='full':
        for i in range(gdfcomid.shape[0]):
            comID = gdfcomid.iloc[i][comIDColName]
            if comID in comIDdict.keys():
                centroid = gdfcomid.iloc[i].geometry.centroid
                #key is col_id
                xv[comIDdict[comID]] = centroid.x
                yv[comIDdict[comID]] = centroid.y            
        G, pos = genGraph(adjMat, comIDdict, xv, yv, graphtype  )                
    else:
        assert (not fpplist is None)
        for i in range(gdfcomid.shape[0]):
            comID = gdfcomid.iloc[i][comIDColName]
            if comID in fpplist:
                centroid = gdfcomid.iloc[i].geometry.centroid
                xv[comIDdict[comID]] = centroid.x
                yv[comIDdict[comID]] = centroid.y            

            G, pos = genGraph(adjMat, comIDdict, fpplist, xv, yv, graphtype, fpplist)

    node_bn = np.array(list(netx.betweenness_centrality(G,normalized=True).values()))
    print (node_bn)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    cmap = plt.get_cmap("inferno")
    norm = matplotlib.colors.Normalize(vmin=np.min(node_bn), vmax=np.max(node_bn), clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    nodecolors=[sm.to_rgba(node_bn[i]) for i in range(len(node_bn))]        
    netx.draw_networkx(G, pos, ax=ax, node_size=25, with_labels=False, edge_color='gray', nodelist=range(len(node_bn)), node_color=nodecolors)
    if not basinboundshp is None:
        #add basin boundary
        basinboundshp = os.path.join(rootdir,basinboundshp)
        boundgdf = gpd.read_file(basinboundshp)
        boundgdf = boundgdf.to_crs('EPSG:4326')
        boundgdf.boundary.plot(ax=ax, alpha=0.8, edgecolor='#626567')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.0)            
    cax.set_title('Betweenness')

    
    ax.set_axis_off()

    plt.colorbar(sm,ax=ax,cax=cax,shrink=0.7)
    plt.savefig(os.path.join(outputdir, 'graphplot.eps'))
    plt.close()


def genGraph(A, comIDdict, xv,yv, graphtype='full', comIDSubset=None):
    """
    graphtype = 'full', 'sub'
    """
    #make sure A and comIDdict have the same order
    G = netx.from_numpy_matrix(A,create_using=netx.DiGraph)

    if graphtype=='sub':
        colInd=[]
        for anode in comIDSubset:
            colInd.append(comIDdict[anode])
        #get subgraph
        subbasinG = G.subgraph(colInd)
        #A and comIDdict have the same order, so colInd have the same order as A order
        #dictionary, key: col number, value: (x,y), 
        pos={colno:(xv[colno],yv[colno]) for ix,colno in enumerate(colInd)}
        return subbasinG,pos
    elif graphtype=='full':
        colInd=comIDdict.values()
                
        #A and comIDdict have the same order, so colInd have the same order as A order
        #dictionary, key: col number, value: (x,y), 
        pos={colno:(xv[colno],yv[colno]) for ix,colno in enumerate(colInd)}
        return G, pos
    

def main():
    #need to call this from 
    assignNodes2Basin(watershed='easttaylor')

if __name__ == '__main__':
    main()
