#alex sun
#https://github.com/selimnairb/EcohydroLib/blob/master/bin/NHDPlusV2Setup/NHDPlusV2Setup.py
#fixed python2.7 bugs
#
import os
import sys
import errno
import subprocess
import sqlite3

from dbf import dbfreader



outputDir = 'nhdplusdb'
pathOfFind='/bin/find'
pathOfSqlite="/work/02248/alexsund/maverick/anaconda3/envs/tensorflow/bin/sqlite3"
pathOfOgr="/work/02248/alexsund/maverick/anaconda3/envs/tensorflow/bin/ogr2ogr"
nhdPlusDB = os.path.join(outputDir, "NHDPlusDB.sqlite")
skipGageLoc = False
skipCatchment = False
skipDB = False
# 1. Find GageLoc shapefile and convert it to a spatial SQLite DB
if not skipGageLoc:
    print("Converting GageLoc shapefile to sqlite database ...")
    gageLocDB = os.path.join(outputDir, "GageLoc.sqlite")
    gageLoc = subprocess.check_output("%s %s -type f -iname GageLoc.shp -print" % (pathOfFind, outputDir,), shell=True).split()
    assert(gageLoc)
    gageLocShp = gageLoc[0].decode('UTF-8') 
    assert(os.access(gageLocShp, os.R_OK))
    print (gageLocShp)
    ogrCommand = '%s -gt 65536 -f "SQLite" -t_srs "EPSG:4326" %s %s' % (pathOfOgr, gageLocDB, gageLocShp)
    print (ogrCommand)
    returnCode = os.system(ogrCommand)
    assert(returnCode == 0) 
    
    # Index fields
    sqliteCommand = "%s %s 'CREATE INDEX IF NOT EXISTS reachcode_measure_idx on GageLoc (reachcode,measure)'" % (pathOfSqlite, gageLocDB)
    returnCode = os.system(sqliteCommand)
    assert(returnCode == 0)
    
    sqliteCommand = "%s %s 'CREATE INDEX IF NOT EXISTS gage_loc_source_fea_idx ON GageLoc (source_fea)'" % (pathOfSqlite, gageLocDB)
    returnCode = os.system(sqliteCommand)
    assert(returnCode == 0)
    #cursor.execute("""""")
    

# 2. Find catchment shapefiles
if not skipCatchment:
    conusCatchment = os.path.join(outputDir, "Catchment.sqlite")
    #print conusCatchment

    # Remove existing conusCatchment
    if os.access(conusCatchment, os.F_OK):
        os.remove(conusCatchment)

    print("Finding catchment shapefiles")
    shapefiles = subprocess.check_output("%s %s -type f -iname Catchment.shp -print" % (pathOfFind, outputDir,), shell=True).split()
    #print shapefiles

    # 3. Intersect all catchment shapefiles into one shapefile for the entire CONUS
    print("Intersecting regional catchment shapefiles in to single CONUS catchment feature dataset ...")
    numFiles = len(shapefiles)
    currFile = 0
    for file in shapefiles:
        file = file.decode('UTF-8') 
        print (file)
        pctComplete = (float(currFile) / float(numFiles)) * 100
        currFile = currFile + 1
        ogrCommand = '%s -gt 65536 -f "SQLite" -append %s %s' % (pathOfOgr, conusCatchment, file)
        #print ogrCommand
        sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)" % (currFile, numFiles, pctComplete))
        sys.stdout.flush()
        returnCode = os.system(ogrCommand)
        assert(returnCode == 0)

    pctComplete = (float(currFile) / float(numFiles)) * 100
    sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)\n" % (currFile, numFiles, pctComplete))

    # 4. Add index to CONUS catchment
    print ("Indexing CONUS shapefile (this may take a while) ...")
    sqliteCommand = "%s %s 'CREATE INDEX IF NOT EXISTS featureid_idx on catchment (featureid)'" % (pathOfSqlite, conusCatchment)
    returnCode = os.system(sqliteCommand)
    assert(returnCode == 0)

# 5. Create NHDPlus SQLite database to store flowline and stream gage records
if not skipDB:
    # Remove existing database if it's there
    if os.access(nhdPlusDB, os.F_OK):
        os.remove(nhdPlusDB)
    #conn = sqlite3.connect(nhdPlusDB, isolation_level=None)
    conn = sqlite3.connect(nhdPlusDB)
    cursor = conn.cursor()
    
    # Create PlusFlowlineVAA table and indices
    cursor.execute("""CREATE TABLE IF NOT EXISTS PlusFlowlineVAA
    (ComID INTEGER,
    Fdate DATETIME,
    StreamLeve INTEGER,
    StreamOrde INTEGER,
    StreamCalc INTEGER,
    FromNode INTEGER,
    ToNode INTEGER,
    Hydroseq INTEGER,
    LevelPathI INTEGER,
    Pathlength REAL,
    TerminalPa INTEGER,
    ArbolateSu REAL,
    Divergence INTEGER,
    StartFlag INTEGER,
    TerminalFl INTEGER,
    DnLevel INTEGER,
    ThinnerCod INTEGER,
    UpLevelPat INTEGER,
    UpHydroseq INTEGER,
    DnLevelPat INTEGER,
    DnMinorHyd INTEGER,
    DnDrainCou INTEGER,
    DnHydroseq INTEGER,
    FromMeas REAL,
    ToMeas REAL,
    ReachCode TEXT,
    LengthKM REAL,
    Fcode INTEGER,
    RtnDiv INTEGER,
    OutDiv INTEGER,
    DivEffect INTEGER,
    VPUIn INTEGER,
    VPUOut INTEGER,
    TravTime INTEGER,
    PathTime INTEGER,
    AreaSqKM REAL,
    TotDASqKM REAL,
    DivDASqKM REAL)
    """)
    cursor.execute("""CREATE INDEX IF NOT EXISTS PlusFlowlineVAA_Comid_idx ON PlusFlowlineVAA (ComID)""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS PlusFlowlineVAA_Reachcode_idx ON PlusFlowlineVAA (ReachCode)""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS PlusFlowlineVAA_FromMeas_idx ON PlusFlowlineVAA (FromMeas)""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS PlusFlowlineVAA_ToMeas_idx ON PlusFlowlineVAA (ToMeas)""")
    
    # Create PlusFlow table and indices
    cursor.execute("""CREATE TABLE IF NOT EXISTS PlusFlow
    (FROMCOMID INTEGER,
    FROMHYDSEQ INTEGER,
    FROMLVLPAT INTEGER,
    TOCOMID INTEGER,
    TOHYDSEQ INTEGER,
    TOLVLPAT INTEGER,
    NODENUMBER INTEGER,
    DELTALEVEL INTEGER,
    DIRECTION INTEGER,
    GAPDISTKM REAL,
    HasGeo TEXT,
    TotDASqKM REAL,
    DivDASqKM REAL)
    """)
    cursor.execute("""CREATE INDEX IF NOT EXISTS plusflow_from_idx ON PlusFlow (FROMCOMID)""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS plusflow_to_idx ON PlusFlow (TOCOMID)""")
    
    # Create Gage_Loc table and indices
    cursor.execute("""CREATE TABLE IF NOT EXISTS Gage_Loc
    (ComID INTEGER,
    EventDate DATETIME, 
    ReachCode TEXT,
    ReachSMDat INTEGER,
    Reachresol TEXT,
    FeatureCom INTEGER,
    FeatureCla INTEGER,
    Source_Ori TEXT,
    Source_Dat TEXT,
    Source_Fea TEXT,
    Featuredet TEXT,
    Measure REAL,
    Offset INTEGER,
    EventType TEXT)
    """)
    cursor.execute("""CREATE INDEX IF NOT EXISTS gage_loc_source_fea_idx ON Gage_Loc (Source_Fea)""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS reachcode_measure_idx on Gage_Loc (ReachCode,Measure)""")
    
    # Create Gage_Info table
    # Gage_Info.GageID maps to Gage_Loc.Source_Fea
    cursor.execute("""CREATE TABLE IF NOT EXISTS Gage_Info
    (GageID TEXT,
    Agency_cd TEXT,
    Station_NM TEXT,
    State_CD TEXT,
    State TEXT,
    DA_SQ_Mile REAL,
    DA_SQ_Km REAL,
    Lon_Site REAL,
    Lat_Site REAL,
    Active TEXT,
    ActiveDate TEXT,
    GagesII TEXT)
    """)
    cursor.execute("""CREATE INDEX IF NOT EXISTS gage_info_gageID_idx ON Gage_Info (GageID)""")
    
    # Create Gage_Smooth table
    # Gage_Smooth.SITE_NO maps to Gage_Info.GageID
    cursor.execute("""CREATE TABLE IF NOT EXISTS Gage_Smooth
    (SITE_NO TEXT,
    YEAR INTEGER,
    MO INTEGER,
    AVE REAL,
    COMPLETERE REAL)
    """)
    cursor.execute("""CREATE UNIQUE INDEX IF NOT EXISTS gage_smooth_idx ON Gage_Smooth (SITE_NO, YEAR, MO)""")
    
    # Create NHDReachCode_Comid table and indices
    cursor.execute("""CREATE TABLE IF NOT EXISTS NHDReachCode_Comid
    (COMID INTEGER,
    REACHCODE TEXT,
    REACHSMDAT DATETIME,
    RESOLUTION TEXT,
    GNIS_ID INTEGER,
    GNIS_NAME TEXT)
    """)
    cursor.execute("""CREATE INDEX IF NOT EXISTS NHDReachCode_Comid_Comid_idx ON NHDReachCode_Comid (COMID)""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS NHDReachCode_Comid_Reachcode_idx ON NHDReachCode_Comid (REACHCODE)""")
    
    # Create NHDFlowline table and indices
    cursor.execute("""CREATE TABLE IF NOT EXISTS NHDFlowline
    (COMID INTEGER,
    FDATE DATETIME,
    RESOLUTION TEXT,
    GNIS_ID INTEGER,
    GNIS_NAME TEXT,
    LENGTHKM REAL,
    REACHCODE TEXT,
    FLOWDIR TEXT,
    WBAREACOMI INTEGER,
    FTYPE TEXT,
    FCODE INTEGER,
    SHAPE_LENG REAL,
    ENABLED TEXT,
    GNIS_NBR INTEGER)
    """)
    cursor.execute("""CREATE INDEX IF NOT EXISTS NHDFlowline_Comid_idx ON NHDReachCode_Comid (COMID)""")
    cursor.execute("""CREATE INDEX IF NOT EXISTS NHDFlowline_Reachcode_idx ON NHDReachCode_Comid (REACHCODE)""")
    
    cursor.close()
    
    # 5. Import NHDPlus data into SQLite database
    # Find PlusFlow.dbf files, open each, import into DB
    print("Importing regional PlusFlowlineVAA.dbf records into CONUS database (this will take a while) ...")
    cursor = conn.cursor()
    dbfs = subprocess.check_output("%s %s -type f -iname PlusFlowlineVAA.dbf -print" % (pathOfFind, outputDir,), shell=True).split()
    numFiles = len(dbfs)
    currFile = 0
    for file in dbfs:
        file = file.decode('UTF-8') 
        print (file)
        #print file
        pctComplete = (float(currFile) / float(numFiles)) * 100
        currFile = currFile + 1
        sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)" % (currFile, numFiles, pctComplete))
        sys.stdout.flush()
        f = open(file, 'rb')
        db = list(dbfreader(f))
        f.close()
        records = db[2:]
        print ('len records', len(records))
        for record in records:
            cursor.execute("""INSERT INTO PlusFlowlineVAA
    (ComID,Fdate,StreamLeve,StreamOrde,StreamCalc,FromNode,ToNode,Hydroseq,LevelPathI,Pathlength,TerminalPa,ArbolateSu,Divergence,StartFlag,TerminalFl,DnLevel,ThinnerCod,UpLevelPat,UpHydroseq,DnLevelPat,DnMinorHyd,DnDrainCou,DnHydroseq,FromMeas,ToMeas,ReachCode,LengthKM,Fcode,RtnDiv,OutDiv,DivEffect,VPUIn,VPUOut,TravTime,PathTime,AreaSqKM,TotDASqKM,DivDASqKM)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record[0], record[1].strftime("%Y-%m-%d %H:%M:%S"), record[2], record[3], 
            record[4], record[5], record[6], record[7], record[8], float(record[9]), 
            record[10], float(record[11]), record[12], record[13], record[14], record[15], 
            record[16], record[17], record[18], record[19], record[20], record[21], 
            record[22], float(record[23]), float(record[24]), str(record[25], 
            errors='replace'), float(record[26]), record[27], record[28], record[29], record[30], record[31], record[32], float(record[33]), float(record[34]), float(record[35]), float(record[36]), float(record[37])))
    conn.commit()
    cursor.close()
    
    pctComplete = (float(currFile) / float(numFiles)) * 100
    sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)\n" % (currFile, numFiles, pctComplete)) 
    
    # Find PlusFlow.dbf files, open each, import into DB
    print("Importing regional PlusFlow.dbf records into CONUS database (this will take a while) ...")
    cursor = conn.cursor()
    dbfs = subprocess.check_output("%s %s -type f -iname PlusFlow.dbf -print" % (pathOfFind, outputDir,), shell=True).split()
    numFiles = len(dbfs)
    currFile = 0
    for file in dbfs:
        file = file.decode('UTF-8') 
        print (file)
        #print file
        pctComplete = (float(currFile) / float(numFiles)) * 100
        currFile = currFile + 1
        sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)" % (currFile, numFiles, pctComplete))
        sys.stdout.flush()
        f = open(file, 'rb')
        db = list(dbfreader(f))
        f.close()
        records = db[2:]
        for record in records:
            #print record
            cursor.execute("""INSERT INTO PlusFlow
    (FROMCOMID,FROMHYDSEQ,FROMLVLPAT,TOCOMID,TOHYDSEQ,TOLVLPAT,NODENUMBER,DELTALEVEL,DIRECTION,GAPDISTKM,HasGeo,TotDASqKM,DivDASqKM)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record[0], record[1], record[2], record[3], record[4], record[5], record[6], record[7], record[8], float(record[9]), record[10], float(record[11]), float(record[12])))
    conn.commit()
    cursor.close()
    
    pctComplete = (float(currFile) / float(numFiles)) * 100
    sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)\n" % (currFile, numFiles, pctComplete)) 
    
    # Find NHDReachCode_Comid.dbf files, open each, import into DB    
    print("Importing regional NHDReachCode_Comid.dbf records into CONUS database (this will take a while) ...")
    cursor = conn.cursor()
    dbfs = subprocess.check_output("%s %s -type f -iname NHDReachCode_Comid.dbf -print" % (pathOfFind, outputDir,), shell=True).split()
    numFiles = len(dbfs)
    currFile = 0
    for file in dbfs:
        file = file.decode('UTF-8') 
        print (file)
        #print file
        pctComplete = (float(currFile) / float(numFiles)) * 100
        currFile = currFile + 1
        sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)" % (currFile, numFiles, pctComplete))
        sys.stdout.flush()
        f = open(file, 'rb')
        db = list(dbfreader(f))
        f.close()
        records = db[2:]
        for record in records:
            #print record
            cursor.execute("""INSERT INTO NHDReachCode_Comid
    (COMID,REACHCODE,REACHSMDAT,RESOLUTION,GNIS_ID,GNIS_NAME)
    VALUES (?,?,?,?,?,?)""",
            (record[0], str(record[1], errors='replace'), record[2].strftime("%Y-%m-%d %H:%M:%S"), str(record[3], errors='replace'), record[4], str(record[5], errors='replace')))
    conn.commit()
    cursor.close()
    
    pctComplete = (float(currFile) / float(numFiles)) * 100
    sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)\n" % (currFile, numFiles, pctComplete)) 
    
    # Find NHDFlowline.dbf files, open each, import into DB 
    print("Importing regional NHDFlowline.dbf records into CONUS database (this will take a while) ...")
    cursor = conn.cursor()
    dbfs = subprocess.check_output("%s %s -type f -iname NHDFlowline.dbf -print" % (pathOfFind, outputDir,), shell=True).split()
  
    numFiles = len(dbfs)
    currFile = 0
    for file in dbfs:
        file = file.decode('UTF-8') 
        print ('dbf ', file)
        #print file
        pctComplete = (float(currFile) / float(numFiles)) * 100
        currFile = currFile + 1
        sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)" % (currFile, numFiles, pctComplete))
        sys.stdout.flush()
        f = open(file, 'rb')
        db = list(dbfreader(f))
        f.close()
        records = db[2:]
        for record in records:
            #print record
            # Handle case where NHDFlowline record lacks GNIS_NBR attribute
            try:
                GNIS_NBR = record[13]
            except IndexError:
                GNIS_NBR = 0
            cursor.execute("""INSERT INTO NHDFlowline
    (COMID,FDATE,RESOLUTION,GNIS_ID,GNIS_NAME,LENGTHKM,REACHCODE,FLOWDIR,WBAREACOMI,FTYPE,FCODE,SHAPE_LENG,ENABLED,GNIS_NBR)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record[0], record[1].strftime("%Y-%m-%d %H:%M:%S"), str(record[2], errors='replace'), record[3], str(record[4], errors='replace'), float(record[5]), str(record[6], errors='replace'), str(record[7], errors='replace'), record[8], str(record[9], errors='replace'), record[10], float(record[11]), str(record[12], errors='replace'), GNIS_NBR))
    conn.commit()
    cursor.close()
    
    pctComplete = (float(currFile) / float(numFiles)) * 100
    sys.stdout.write("\r\tProcessing file %d of %d (%.0f%%)\n" % (currFile, numFiles, pctComplete)) 
    
    # Find GageLoc.dbf file, import into DB 
    print("Importing national GageLoc.dbf ...")
    cursor = conn.cursor()
    dbf = subprocess.check_output("%s %s -type f -iname GageLoc.dbf -print" % (pathOfFind, outputDir,), shell=True).split()[0]
    assert(dbf)
    print (dbf)
    f = open(dbf, 'rb')
    db = list(dbfreader(f))
    f.close()
    records = db[2:]
    for record in records:
        #print record
        cursor.execute("""INSERT INTO Gage_Loc
    (ComID,EventDate,ReachCode,ReachSMDat,Reachresol,FeatureCom,FeatureCla,Source_Ori,Source_Dat,Source_Fea,Featuredet,Measure,Offset,EventType)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (record[0], record[1].strftime("%Y-%m-%d %H:%M:%S"), str(record[2], errors='replace'), record[3], str(record[4], errors='replace'), record[5], record[6], str(record[7], errors='replace'), str(record[8], errors='replace'), str(record[9], errors='replace'), str(record[10], errors='replace'), float(record[11]), float(record[12]), str(record[13], errors='replace')))
    conn.commit()
    cursor.close()
    
    # Find GageInfo.dbf file, import into DB 
    print("Importing national GageInfo.dbf ...")
    cursor = conn.cursor()
    dbf = subprocess.check_output("%s %s -type f -iname GageInfo.dbf -print" % (pathOfFind, outputDir,), shell=True).split()
    assert(dbf)
    dbf = dbf[0].decode('UTF-8')     
    print (dbf)
    f = open(dbf, 'rb')
    db = list(dbfreader(f))
    f.close()
    records = db[2:]
    for record in records:
        #print record
        # Handle presence of undocumented NHD2DAGE_D field (if present)
        if len(record) > 12:
            cursor.execute("""INSERT INTO Gage_Info
        (GageID,Agency_cd,Station_NM,State_CD,State,SiteStatus,DA_SQ_Mile,Lon_Site,Lat_Site,Lon_NHD,Lat_NHD,Reviewed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(record[0],errors='replace'), str(record[1], errors='replace'), str(record[2], errors='replace'), str(record[3], errors='replace'), str(record[4], errors='replace'), str(record[5], errors='replace'), float(record[6]), float(record[7]), float(record[8]), float(record[9]), float(record[10]), str(record[12], errors='replace') ))
        else:
            cursor.execute("""INSERT INTO Gage_Info
        (GageID,Agency_cd,Station_NM,State_CD,State,DA_SQ_Mile,DA_SQ_Km,Lat_Site,Lon_Site,Active,ActiveDate,GagesII)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(record[0],errors='replace'), str(record[1], errors='replace'), 
            str(record[2], errors='replace'), str(record[3], errors='replace'), 
            str(record[4], errors='replace'), 
            float(record[5]), float(record[6]), 
            float(record[7]), float(record[8]), 
            str(record[9], errors='replace'), 
            str(record[10], errors='replace'),
            str(record[11], errors='replace') ))
    conn.commit()
    cursor.close()
    
    # Find Gage_Smooth.DBF file, import into DB 
    print("Importing national Gage_Smooth.DBF ...")
    cursor = conn.cursor()
    dbf = subprocess.check_output("%s %s -type f -iname Gage_Smooth.DBF -print" % (pathOfFind, outputDir,), shell=True).split()[0]
    assert(dbf)
    dbf = dbf.decode('UTF-8')
    print (dbf)
    f = open(dbf, 'rb')
    db = list(dbfreader(f))
    f.close()
    records = db[2:]
    for record in records:
        #print record
        cursor.execute("""INSERT INTO Gage_Smooth
    (SITE_NO,YEAR,MO,AVE,COMPLETERE)
    VALUES (?,?,?,?,?)""",
        (str(record[0], errors='replace'), record[1], record[2], float(record[3]), float(record[4]) ))
    conn.commit()
    cursor.close()
    
    conn.close()