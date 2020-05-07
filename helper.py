import numpy as np
import torch

INPUT = 6
OUTPUT = 10
t_h = INPUT  # length of track history#Ben: Input size (History)
t_f = OUTPUT  # length of predicted trajectory#Ben: output size (Future)
d_s = 1  # down sampling rate of all sequences# TODO#내가 만약 history를 6개를 보는데 d_s가 2라면 결국 띄엄띄엄해서 3개의 history를 보는 것.
enc_size = 64 # size of encoder LSTM
grid_size = (13,3) # size of social context grid
upp_grid_size = (7,3)
inds = [14,15,16,17,18,19,20, 27,28,29,30,31,32,33, 40,41,42,43,44,45,46]# HORIZON; front Info.
#-->left top + center top + right top//BEN
dwn_inds = [8,9,10,11,12,13, 21,22,23,24,25,26, 34,35,36,37,38,39]# NEIGHBOR?; rear Info.
#-->left bottom + center bottom + right bottom//BEN

def getHistory(vehId, t, refVehId, dsId, current, 
                track_data = None, t_h = t_h, t_f = t_f):
    # 결국, refVeh의 t에서의 x,y를 기준으로, veh의 t부터 t_h까지의 상대적인 x,y의 history
    if not vehId in track_data[dsId].keys():
        # print(vehId, track_data[dsId].keys())
        return current, np.array([0, 0]), False
    
    refTrack = (track_data[dsId][refVehId].transpose()).astype(float)
    vehTrack = (track_data[dsId][vehId].transpose()).astype(float)
    
    # vehId의 refVehId가 같은 frame에 있었을 때의 x,y를 수집
    refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]#Ben: Other objects' pose at time t (at that frame)

    if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
        return current, refPos, False#0
    else:
        #현재t보다 t_h뒤의 index. 만약, 현재 t기준으로 t_h 뒤가 trajectory 초기의 t값 wmr, t-t_h <0이면, 그때는 0의 값을 갖는다.
        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item(0) - t_h)
        #현재t보다 하나 더 앞의 index
        enpt = np.argwhere(vehTrack[:, 0] == t).item(0) + 1
        hist = vehTrack[stpt:enpt:1,1:3]-refPos#Normalized?
        #vehTrack에서 현재 stpt부터 enpt까지의 x,y
    if len(hist) < t_h//1 + 1:#TODO.... weird..
        # if vehId == refVehId:
        #     print("len(hist) < t_h+1: ", current, refPos)
        return current, refPos, False
    
    #     print("vehID === REF")
    #     print(vehTrack[stpt:enpt:1,1:3])

                                                                                                    # Behavioral Modification 3: Change inputs
    m1 = int(t_h/3)#Ben: length of history / 3; 1/3지점
    m2 = 2 * m1#2/3지점,,,,, t_h는 3/3지점.
    vel0 = np.array([[hist[m1][0] - hist[0][0], hist[m1][1] -hist[0][1]]])#현재를 기준으로 우리가 보는 history길이의 1/3 전의 x,y
    vel5 = np.array([[hist[m2][0] - hist[m1][0], hist[m2][1] -hist[m1][1]]])#현재를 기준으로 우리가 보는 history길이의 2/3 전의 x,y
    vel10 = np.array([[hist[t_h][0] - hist[m2][0], hist[t_h-1][1] -hist[m2][1]]])#현재를 기준으로 우리가 보는 history길이의 끝에서의 x,y
    hist = np.concatenate((hist, np.concatenate((vel0,vel5,vel10), axis=0)), axis=0)# velocity?
    return hist, refPos, True

def getFuture(vehId, t,dsId, track_data = None, d_s = d_s,
                                   t_h = t_h, t_f = t_f):

    vehTrack = (track_data[dsId][vehId].transpose()).astype(float)
    refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
    stpt = np.argwhere(vehTrack[:, 0] == t).item(0) + d_s
    enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item(0) + t_f + 1)
    fut = vehTrack[stpt:enpt:d_s,1:3]-refPos#Ben: Based on the current pose, get future gt trajectory
    return fut

def PreprocessDataset(traj_data_t, track_data, inds = [14,15,16,17,18,19,20, 27,28,29,30,31,32,33, 40,41,42,43,44,45,46],
                                    dwn_inds = [8,9,10,11,12,13, 21,22,23,24,25,26, 34,35,36,37,38,39],
                                    d_s = d_s, t_h = t_h, t_f = t_f, enc_size = enc_size,
                                    grid_size = grid_size, upp_grid_size = upp_grid_size):
    '''traj_data_t means all data from 'data_idx' which are in the same "frame_idx"'''
    samples = list()
    l_refPos = list()
    hist_enough = list()
    # for idx in list(indices):
    for data_row in traj_data_t:
        # traj file
        dsId = data_row[0].astype(int)# dsId.txt
        vehId = data_row[1].astype(int)# unique Vehicle ID in dataset
        t = data_row[2] # Frame
        current = np.array([data_row[3:5]])# x, y
        
        grid = data_row[8:47] # Ben: surrounding obstacles using distance# Neighbor! TODO
        upp_grid = data_row[inds]# Inner-distance of forward objects# Horizon! TODO
        neighbors = []
        upper_neighbors = []
        hist, refPos, histfull = getHistory(vehId,t,vehId,dsId,current, track_data=track_data)#현재(t) 자신(vehId)의 x,y를 기준으로 과거(t부터t_h까지)의 상대적인 dx, dy
        l_refPos.append(refPos)
        hist_enough.append(histfull)
        fut = getFuture(vehId,t,dsId, track_data=track_data)#현재(t) 자신(vehId)의 x,y를 기준으로 미래(t부터t_f까지)의 상대적인 dx, dy

        for i in grid:#NEIGHBOR's idx
            #현재(t) 자신(vehId)의 x,y를 기준으로 Neighbor들의 과거(t부터t_h까지)의 상대적인 dx, dy
            neigh_hist, _, _ = getHistory(i.astype(int), t,vehId,dsId, current, track_data=track_data)
            neighbors.append(neigh_hist)
                                                                                        #Behavioral Modification 2: Adding Kinetic Flow layer
            #현재(t) 자신(vehId)의 x,y를 기준으로 Horizon들의 과거(t부터t_h까지)의 상대적인 dx, dy
        for i in upp_grid:#HORIZON's idx
            hori_hist, _, _ = getHistory(i.astype(int), t,vehId,dsId, current, track_data=track_data)
            upper_neighbors.append(hori_hist)

        upp_count = np.count_nonzero(upp_grid)# front objects의 개수
        dwn_count = np.count_nonzero(data_row[dwn_inds])#rear objects의 개수
        hist = np.concatenate((hist, np.array([[upp_count, dwn_count]])), axis=0)#len(hist) == t_h+3+1(?)
        # print("hist: ", hist)
        lon_enc = np.zeros([2])
        lon_enc[int(data_row[7] - 1)] = 1#lon_enc = [0, 1]
        lat_enc = np.zeros([3])
        lat_enc[int(data_row[6] - 1)] = 1#lat_enc = [0, 0, 1]

        samples.append((hist,fut,upper_neighbors, neighbors,lat_enc,lon_enc, False, dsId, vehId, t))
    
    ##### ===== same with collate_fn ===== #####  
    nbr_batch_size = 0    
    for _, _, upp_nbrs, nbrs, _, _, _, _, _, _ in samples:
        nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
    maxlen = t_h//d_s + 1+4# Behavioral Modification 3: Change inputs/ change max len to +3
    nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)

    upp_nbr_batch_size = 0
    for _, _, upp_nbrs, nbrs, _, _, _, _, _, _ in samples:
        upp_nbr_batch_size += sum([len(upp_nbrs[i])!=0 for i in range(len(upp_nbrs))])
    upp_maxlen = t_h//d_s + 1+3                                                               # Behavioral Modification 3: Change inputs/ change max len to +3
    upp_nbrs_batch = torch.zeros(upp_maxlen,upp_nbr_batch_size,2)

    # Initialize social mask batch:#grid_size = (13,3), upp_grid_size = (7,3)
    pos = [0, 0]
    mask_batch = torch.zeros(len(samples), grid_size[1], grid_size[0], enc_size)
    mask_batch = mask_batch.byte()

    upp_pos = [0,0]                                                                                 # Behavioral Modification 2: Adding Kinetic Flow layer
    upp_mask_batch = torch.zeros(len(samples), upp_grid_size[1], upp_grid_size[0], enc_size)
    upp_mask_batch = upp_mask_batch.byte()

    # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
    # len(samples) <-- batch size
    hist_batch = torch.zeros(maxlen,len(samples),2)
    fut_batch = list()
    op_mask_batch = torch.zeros(t_f//d_s,len(samples),2)
    lat_enc_batch = torch.zeros(len(samples),3)
    lon_enc_batch = torch.zeros(len(samples), 2)

    count = 0
    upp_count = 0
    
    sampleId = 0

    for sampleId,(hist, fut, upp_nbrs, nbrs, lat_enc, lon_enc, bb, dd, vv, ff) in enumerate(samples):
        
        hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])   #x
        hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1]) #y
        fut_batch.append(fut)
        op_mask_batch[0:len(fut),sampleId,:] = 1# future masking?
        lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
        lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
        
        # Set up neighbor, neighbor sequence length, and mask batches:
        for id,nbr in enumerate(nbrs):
            if len(nbr)!=0:#if there is a history of nbrs
                nbrs_batch[0:len(nbr), count,0] = torch.from_numpy(nbr[:, 0])#x
                nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])#y
                pos[0] = id % grid_size[0]#13
                pos[1] = id // grid_size[0]#13
                mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(enc_size).byte()
                count+=1#each count, each neighbor

        for id, upp_nbr in enumerate(upp_nbrs):
            if len(upp_nbr) != 0:
                upp_nbrs_batch[0:len(upp_nbr), upp_count, 0] = torch.from_numpy(upp_nbr[:, 0])
                upp_nbrs_batch[0:len(upp_nbr), upp_count, 1] = torch.from_numpy(upp_nbr[:, 1])
                upp_pos[0] = id % upp_grid_size[0]#7
                upp_pos[1] = id // upp_grid_size[0]
                upp_mask_batch[sampleId, upp_pos[1], upp_pos[0], :] = torch.ones(enc_size).byte()#byte() just declares
                upp_count += 1

    return vehId, fut_batch, l_refPos, hist_enough, hist_batch.cuda(), upp_nbrs_batch.cuda(), nbrs_batch.cuda(), upp_mask_batch.cuda(),\
           mask_batch.cuda(), lat_enc_batch.cuda(), lon_enc_batch.cuda()