''' This main.py is for comparison of trajectory prediction method'''
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import graph.sample
# import traphic.sample
from utils import DataLoader
from traphic.model.model import TnpModel
from helper import *

def main():
    '''main code'''
    parser = argparse.ArgumentParser()

    # NOTE: argument for comparison of algorithms
    parser.add_argument(
        "--manual_seed",
        type=int,
        default=42,
        help="Random seed for torch & random function",
    )
    parser.add_argument(
        "--obs_length",
        type=int,
        default=6,
        help="History length of the trajectory"
    )
    parser.add_argument(
        "--pred_length",
        type=int,
        default=10,
        help="predicted length of the trajectory"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='./data/prediction_train/',
        help="Data directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='APOL',
        help="Which dataset is used"
    )
    args = parser.parse_args()
    random_seed = args.manual_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    ''' get_data '''
    # generate data samples
    dataloader = DataLoader(args, using_testset=True)
    print("current dataset is {}".format(dataloader.dataset_name[0]))
    dataset_idx = 0
    frame_idx = 5#Interesting frame; must be larger than 'obs_length-1'
    dataset_name = dataloader.dataset_name[dataset_idx]
    x_input, frame_IDs = dataloader.get_sample(dataset_idx, frame_idx)
    scale_param = dataloader.scale_param
 
    #4d_graph_
    # graph.sample.main(x_input, frame_IDs, scale_param)

    ########################################## TRAPHIC ################################################
    DATASET = 'APOL'
    print("Using {} dataset...".format(DATASET))
    # LOAD = 'Traphic_CARLA_model_10-10l_200e_using.CARLA.dataset.tar'
    LOAD = 'Traphic.APOL.model_6-10l_50epochs.tar'
    MODELLOC = "./save_weight_"
    DATA_DIR =  './data/' + DATASET
    RAW_DATA = "./data/prediction_train/"#Ben: APOL

    LOG = './logs/'
    DEVICE = 'cuda:0'
    INPUT = 6
    OUTPUT = 10
    PRETRAINEPOCHS = 10
    TRAINEPOCHS = 90
    # NAME = '{}_{}' + '_model_{}-{}l_{}e_using.{}.dataset.tar'.format(INPUT, OUTPUT, PRETRAINEPOCHS + TRAINEPOCHS, DATASET)
    NAME = LOAD
    dtype = 'test'
    traj_dir = "./data/{}/{}/{}Set0-traj.npy".format(DATASET, dtype, dtype)
    track_dir = "./data/{}/{}/{}Set0-track.npy".format(DATASET, dtype,dtype)
    traj_data = np.load(traj_dir, allow_pickle=True)[0]
    track_data = np.load(track_dir, allow_pickle=True)[0]#dict

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

    viewArgs = {}
    viewArgs['cuda'] = True
    viewArgs['log_dir'] = LOG
    viewArgs['batch_size'] = 1
    viewArgs['dropout'] = 0.5
    viewArgs["lr"] = 0.001
    viewArgs["optim"] = 'Adam'
    viewArgs['pretrainEpochs'] = PRETRAINEPOCHS
    viewArgs['trainEpochs'] = TRAINEPOCHS
    viewArgs["maneuvers"] = False
    viewArgs['predAlgo'] = 'Traphic'
    viewArgs['pretrain_loss'] = 'NLL'
    viewArgs['train_loss'] = 'MSE'
    viewArgs['tensorboard'] = False
    viewArgs['modelLoc'] = MODELLOC
    viewArgs['dir'] = DATA_DIR
    viewArgs['raw_dir'] = RAW_DATA
    viewArgs['device'] = DEVICE
    viewArgs['dsId'] = 0#Ben: 0 for prediction-{train,test}.zip and 1 for sample dataset
    viewArgs['dset'] = DATASET
    viewArgs['name_temp'] = NAME#Ben: Model name!
    viewArgs['input_size'] = INPUT#Ben: # of input sequence?
    viewArgs['output_size'] = OUTPUT#Ben: # of output sequence?

    model = TnpModel(viewArgs)
    model.load(LOAD)

    # Get relevant vehicles
    print("current dataset_name: ", dataset_name)
    
    # Get all dataset_idx.txt which corresponds to 'that dataset'
    traj_data_dsId = traj_data[traj_data[:, 0]==int(dataset_name)]
    # _dsId = traj_data[dataset_idx, 0].astype(int)#Get all dataset_idx.txt
    traj_data_t = traj_data_dsId[traj_data_dsId[:, 2]==frame_idx] # Extract all data of current frame

    print("first row: ", traj_data_t[0])
    
    vehid, fut, l_refPos, hist_enough, hist_batch, upp_nbrs_batch, nbrs_batch, upp_mas_batch, mask_batch, lat_enc_batch, lon_enc_batch = PreprocessDataset(traj_data_t, track_data)
    history_indices = hist_batch.cpu().detach().numpy()
    history_indices = np.swapaxes(history_indices,0,1)

    fut_pred_info = model.net(hist_batch, upp_nbrs_batch, nbrs_batch, upp_mas_batch, mask_batch, lat_enc_batch, lon_enc_batch)
    
    fut_pred_trj = fut_pred_info.cpu().detach().numpy()
    fut_pred_trj = np.swapaxes(fut_pred_trj, 0, 1)
    
    for k in range(len(history_indices)):
        if hist_enough[k] == True:
            past_trj = history_indices[k,:INPUT,:]
            past_trj = np.reshape(past_trj, (INPUT,2))
            trj_x = list(past_trj[:,0]+l_refPos[k][0])
            trj_y = list(past_trj[:,1]+l_refPos[k][1])
        else:
            past_trj = history_indices[k,:1,:]
            past_trj = np.reshape(past_trj, (1,2))
            trj_x = list(past_trj[:,0])
            trj_y = list(past_trj[:,1])
        
        gt_x = np.array([trj_x[-1]] + list(fut[k][:,0]+l_refPos[k][0]))
        gt_y = np.array([trj_y[-1]] + list(fut[k][:,1]+l_refPos[k][1]))
        ft_x = np.array([trj_x[-1]] + list(fut_pred_trj[k,:fut[k].shape[0],0]+l_refPos[k][0]))
        ft_y = np.array([trj_y[-1]] + list(fut_pred_trj[k,:fut[k].shape[0],1]+l_refPos[k][1]))

        ##### METRIC #####
        x_diff = (gt_x - ft_x)**2
        y_diff = (gt_y - ft_y)**2
        dis_diff = np.sqrt(x_diff + y_diff)#distance^2
        ADE = np.mean(dis_diff)
        FDE = dis_diff[-1]
        RMSE = np.sqrt(sum(x_diff + y_diff)/dis_diff.shape[0])
        plt.title("ADE: {}, FDE: {}, RMSE: {}".format(round(ADE,3), round(FDE,3), round(RMSE,3)))
        
        # xmin = min(min(trj_x), min(gt_x))
        # xmax = max(max(trj_x), max(gt_x))
        # ymin = min(min(trj_y), min(gt_y))
        # ymax = max(max(trj_y), max(gt_y))        
        # # xmin, xmax = -30, 30
        # # ymin, ymax = -30, 30
        # plt.xticks(np.arange(xmin, xmax, 0.5))
        # plt.yticks(np.arange(ymin, ymax, 0.5))


        textstr = '\n'.join(('ADE={}'.format(round(ADE,3)),'FDE={}'.format(round(FDE,3))))
        # plt.text(0.05, 0.95, textstr,  fontsize=14)

        print(textstr)
        plt.plot(trj_x, trj_y, 'ko--')
        plt.plot(gt_x, gt_y, 'g->')
        plt.plot(ft_x, ft_y, 'r->')
        plt.axis('equal')
        # plt.grid() 
    plt.show()

if __name__ == "__main__":
    main()