# Houston GNN project

This repo contains files related to Houston GNN study

### Data preparation
1. NHDPlusDB, download tar and put to some directory and modify configuration file
2. Put the pkl files under data/houston directory


### Running the codes

In the following, Step 1 prepares river network, Step 2 prepares datasets, and Step 3 trains/tests the network model
1. >. rungenriver
2. >. runloadriver
   
Make sure ./models/houston exist, then run

3. >. rungwnet

### Outputs from step 1
```
Loading saved NWM data
cutoff 0.001
----Tree Cutoff Threshold is Median 0.001 m3/s
length 518 518
shape of nwmdf before trimming (9496, 518)
shape of nwmdf after trimming (9496, 470)
gage comid set [1440385, 1440389, 1440277, 1440237, 1439317, 1439357, 1440183]
Warning: getStaticData was not successful
```

### Outputs from step 2
```
Loading saved NWM data
cutoff 0.001
----Tree Cutoff Threshold is Median 0.001 m3/s
length 518 518
shape of nwmdf before trimming (9496, 518)
shape of nwmdf after trimming (9496, 470)
gage comid set [1440385, 1440389, 1440277, 1440237, 1439317, 1439357, 1440183]
Warning: getStaticData was not successful
adj mat index of the observation nodes [356, 357, 331, 321, 206, 214, 308]
doing log transform
Time taken to form input data 10.920103525742888
train data shape torch.Size([6617, 30, 470, 3]) torch.Size([6617, 470])
val data shape torch.Size([1424, 30, 470, 3]) torch.Size([1424, 470])
test data shape torch.Size([1425, 30, 470, 3]) torch.Size([1425, 470])
```

### Partial outputs from step 3
```
....
# Epoch 29: 100%|████████████████████████████████████████████████████████████████████████████████████| 220/220 [02:17<00:00,  1.60it/s]
time elapsed  137.1682412624359
epoch 29 , train loss: 0.001279912373107491 , val loss: 0.0013829248554423346
use saved best model  .../models/houston/gwnet2bestmodel_seq30_L1_seed4121046_node470_nwm2.0_aorc_hourly_usenwm_log.pth
Get testing results ...
test mat shape (1425, 470)
median nse 0.237, mean nse 0.279, max nse 0.924, min nse -0.202
Loading saved NWM data
cutoff 0.001
----Tree Cutoff Threshold is Median 0.001 m3/s
length 518 518
shape of nwmdf before trimming (9496, 518)
shape of nwmdf after trimming (9496, 470)
gage comid set [1440385, 1440389, 1440277, 1440237, 1439317, 1439357, 1440183]
Warning: getStaticData was not successful
in plotting NSE,  COMID
in plotting, number of comids  470
         COMID       NSE
0    1438179.0  0.173757
1    1438185.0  0.204069
2    1438187.0  0.114199
3    1438191.0  0.191629
4    1438197.0  0.160923
..         ...       ...
465  1560080.0  0.203687
466  1560082.0  0.178553
467  1560084.0  0.199825
468  1562186.0  0.244378
469  1562208.0  0.431059

[470 rows x 2 columns]
(470, 2)
     COMID                                           geometry       NSE
0  1438269  LINESTRING Z (-95.26841 29.95451 0.00000, -95....  0.222846
1  1439759  LINESTRING Z (-95.42170 29.61992 0.00000, -95....  0.226454
2  1559698  LINESTRING Z (-95.01373 29.81171 0.00000, -95....  0.231321
3  1440269  LINESTRING Z (-95.50265 29.79692 0.00000, -95....  0.129999
4  1438855  LINESTRING Z (-95.40283 29.84994 0.00000, -95....  0.172974
Index(['COMID', 'geometry', 'NSE'], dtype='object')
```
