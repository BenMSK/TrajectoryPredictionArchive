import numpy as np
import random
import os
import pickle
import csv

from traphic.model.import_data import *
RAW_DATA = "./data/prediction_train/"#Ben: APOL

class DataLoader():

    def __init__(self, args, using_testset=False):
        """
        It uses APOL dataset!NOTE currently...
        Dataset means one single logging .txt or .csv file.
        seq_length : Sequence length to be considered  21
        datasets : The indices of the datasets to use
        """
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        self.using_testset = using_testset

        # List of data directories where raw data resides
        self.data_dirs = RAW_DATA#"./data/prediction_train/"
        self.dataset_cnt = len(os.listdir(self.data_dirs))# Ben: Get the number of all data in 'data_dirs'
        self.datasets_dir = sorted(os.listdir(self.data_dirs))# Ben: Sort the data name by alphabet order
        np.random.shuffle(self.datasets_dir)# Shuffle the training data

        # Store the arguments
        self.obs_length = args.obs_length
        self.pred_length = args.pred_length
        self.seq_length = args.obs_length + args.pred_length

        # (training, validation, test) = (0.7, 0.2, 0.1)
        if using_testset == True:
            self.datasets_dir = self.datasets_dir[int(self.dataset_cnt * 0.9) :]
            data_file = os.path.join("./data/", "test_trajectories-{}.cpkl".format(args.manual_seed))# only has 10% of training.txt

        # If the file doesn't exist or forcePreProcess is true
        if not (os.path.exists(data_file)):
            print("Creating pre-processed data from raw data")# .ckpl file is generated
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.generate_pkl_file(self.datasets_dir, data_file)

        # Load the processed data from the pickle file
        self.load_pkl_file(data_file)


        #### Traphic
        generate_data = False
        if generate_data:
            threadid = 1
            print('dataset for TraPHic is generated...')
            files = None
            train_loc = RAW_DATA
            output_dir = RAW_DATA + 'test_obs/formatted/'
            files = self.datasets_dir
            train_lst = self.apol_to_formatted(train_loc, files, output_dir, "test")
            npy_loc = './data/APOL' 
            self.create_data(output_dir, train_lst, npy_loc, "test", threadid)

        

    def get_sample(self, dataset_idx, frame_idx):
        """
        data = [[[frame1 data],     #first dataset// only for training; valid_data has same form.
                 [frame2 data],
                      ....   ]],
                [[frame1 data],     #second dataset
                 [frame2 data],
                      ....   ]]]
        frameList = [[1st f, 2nd f, 3rd f, ...],     #first dataset
                     [1st f, 2nd f, 3rd f, ...],     #second dataset
                            ....,
                     [1st f, 2nd f, 3rd f, ...]]     #last dataset
        
        numPedsList = [[# of agents in 1st f, # of agents in 2nd f, # of agents in 3rd f, ...],     #first dataset
                       [# of agents in 1st f, # of agents in 2nd f, # of agents in 3rd f, ...],     #second dataset
                            ....,
                       [# of agents in 1st f, # of agents in 2nd f, # of agents in 3rd f, ...]]     #last dataset
        Function to get the next batch of points
        """
        all_frame_data = self.data[dataset_idx]
        print("Test data from dataset idx {}: {}"\
              .format(dataset_idx, self.dataset_name[dataset_idx])
            )
        # Increment the counter with the number of sequences in the current dataset
        # I.E. count the number of sequences for training

        # Source data
        x_batch = []
        # Frame data
        frame_batch = []

        # Extract the frame data of the current dataset
        # Mini-batch만큼 설정.
        frame_data = self.data[dataset_idx]# Get all data of 000 dataset
        frame_ids = self.frameList[dataset_idx]# Get frame list of '#.txt dataset'
        # Get the frame pointer for the current dataset
        frame_src_point = frame_idx+1 - self.obs_length
        if frame_src_point + self.seq_length < len(frame_data):
            # All the data in this sequence
            # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
            seq_source_frame_data = frame_data[frame_src_point : frame_src_point + self.seq_length]# index부터 history 길이만큼의 frame data를 모두 가져온다.   ex) [0, 1, 2, 3, ...]
            seq_frame_ids = frame_ids[frame_src_point : frame_src_point + self.seq_length]

            # Number of unique peds in this sequence of frames
            x_batch.append(seq_source_frame_data)
            frame_batch.append(seq_frame_ids)
        else:
            print("too short for prediction")
            # Not enough frames left
            # Increment the dataset pointer and set the frame_pointer to zero

        return x_batch, frame_batch


    def generate_pkl_file(self, data_dirs, data_file):
        """
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        """
        # all_frame_data would be a list of list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
        # containing pedID, x, y
        all_frame_data = []# [ #dataset x #frame x (#numPeds x 3) ]
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []# [ #dataset x #frame ]
        # numAgents would be a list of lists corresponding to each dataset
        # Each list would contain the number of pedestrians in each frame in the dataset
        numAgents = []# [ #dataset x #frame ]
        dataset_name = []
        # Index of the current dataset
        dataset_index = 0

        # Initialization// currently we use the scale parameter of prediction_train file
        min_position_x = 1000#0.389#
        max_position_x = -1000#708.908#
        min_position_y = 1000#0.674#
        max_position_y = -1000#348.585#


        for ind_directory, directory in enumerate(data_dirs):# Get min/max position along all datasets
            file_path = os.path.join(RAW_DATA, directory)
            data = np.genfromtxt(file_path, delimiter=" ")
            min_position_x = min(min_position_x, min(data[:, 3]))
            max_position_x = max(max_position_x, max(data[:, 3]))
            min_position_y = min(min_position_y, min(data[:, 4]))
            max_position_y = max(max_position_y, max(data[:, 4]))
        scale_list = [min_position_x, min_position_y, max_position_x, max_position_y]
        
        # For each dataset
        for ind_directory, directory in enumerate(data_dirs):
            # define path of the csv file of the current dataset
            # file_path = os.path.join(directory, 'pixel_pos.csv')
            splitted_name = directory.split('_')
            dset_name = splitted_name[1]+splitted_name[2].zfill(2)
            dataset_name.append(dset_name)
            
            file_path = os.path.join(RAW_DATA, directory)# Each .txt or .csv file

            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=" ")
            # print(min_position_x)
            """
            scaling each position;
            If you want to get real difference; you should scale up all data set again
            or save each scale rate for each dataset and multiply them, later,
            Scale range: [-1.0, +1.0]
            """
            data[:, 3] = (#3 for training
                (data[:, 3] - min_position_x) / (max_position_x - min_position_x)
            ) * 2 - 1 # Scale range [-1, 1]
            data[:, 4] = (#4
                (data[:, 4] - min_position_y) / (max_position_y - min_position_y)
            ) * 2 - 1

            data = data[~(data[:, 2] == 5)]#Ben: DO NOT CONSIDER 'others' OBJECT. [REMOVE]#sample_: 4

            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[:, 0]).tolist()
            numFrames = len(frameList)# Total frame

            # Add the list of frameIDs to the frameList_data 'for each .txt file'
            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numAgents.append([])
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])

            for ind, frame in enumerate(frameList):
                # Extract all agents in current frame
                agentsInFrame = data[data[:, 0] == frame, :]

                # Extract agents list
                agentsList = agentsInFrame[:, 1].tolist()# Get the pedsID

                # Add number of peds in the current frame to the stored data
                numAgents[dataset_index].append(len(agentsList))

                # Initialize the row of the numpy array
                agentsWithPos = []
                # For each ped in the current frame
                for agent in agentsList:#get all agent' Info in the current frame
                    # Extract their x and y positions
                    current_x = agentsInFrame[agentsInFrame[:, 1] == agent, 3][0]
                    current_y = agentsInFrame[agentsInFrame[:, 1] == agent, 4][0]
                    current_type = self.class_objtype(int(agentsInFrame[agentsInFrame[:, 1] == agent, 2][0]))# Object type
                    # Add their agentID, x, y to the row of the numpy array
                    agentsWithPos.append([agent, current_x, current_y, current_type])

                all_frame_data[dataset_index].append(np.array(agentsWithPos))
            
            dataset_index += 1

        # Save the tuple (all_frame_data, frameList_data, numAgents) in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numAgents, scale_list, dataset_name), f, protocol=2)
        f.close()
        #all_frame_data: 각 데이터 .txt 파일에서, current frame에서의 agents' Info.
        #                --> = [ [ [0번 frame agents'Info], [1번 frame agents'Info], ... ] ], #1.txt파일
        #                        [ [0번 frame agents'Info], [1번 frame agents'Info], ... ] ], #2.txt파일
        #                                            ......
        #frameList_data:  각 데이터 .txt 파일에 존재하는 frame. = [1.txt의 frame id들, 2.txt의 frame id들, .... ]
        #numAgents:  각 데이터 .txt 파일에서, 그 파일의 어느 frame에 존재하는 ped의 수. 
        #               --> = [ [1.txt의 0번 frame의 agents 수, 1.txt의 1번 frame의 agents 수, ... ],
        #                       [2.txt의 0번 frame의 agents 수, 2.txt의 1번 frame의 agents 수, ... ],
        #                                       .....                                          ]
    
    def load_pkl_file(self, data_file):
        ''' Load data from the pickled file '''
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()
        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.scale_param = self.raw_data[3]#NOTE: Ben
        self.dataset_name = self.raw_data[4]

        # print("Scale parameter: ", self.scale_param)


    def class_objtype(self, object_type):#논문에서 c_i
        if object_type == 1 or object_type == 2:#Vehicle
            return 3
        elif object_type == 3:#Pedestrian
            return 1
        elif object_type == 4:#Bicycle
            return 2
        else:
            return -1

    def apol_to_formatted(self, input_dir, files, output_dir, dtype):
        txtlst = []
        i = 0 
        sz = len(files)
        print("=======================================")
        for f in files:
            print("Processing {}/{} in {}...".format(i, sz, dtype))
            # print("files: ", f)
            i += 1
            splitted_name = f.split('_')
            dset_id = splitted_name[1] + splitted_name[2].zfill(2)
            
            out_name = dset_id + '.txt'
            txtlst.append(dset_id)
            
            current_time = -1
            current_frame_num = -1

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            out = open(os.path.join(output_dir, out_name), 'w')
            f = os.path.join(input_dir, f)

            with open(f) as csv_file:
                for row in csv.reader(csv_file):
                    # print('row: ', row)
                    if row[0] == 'TIMESTAMP':
                        continue;
                    each_row = row[0].split(' ')
                    # print('row: ', each_row)
                    if float(each_row[0]) > current_time:
                        current_time = float(row[0][0])
                        current_frame_num += 1
                    # print(each_row[1])
                    vid = int(each_row[1].split('-')[-1])
                    # out.write("{},{},{},{},{}\n".format(float(dset_id), vid, current_frame_num, each_row[2], each_row[3]))
                    out.write("{},{},{},{},{}\n".format(float(dset_id), vid, current_frame_num, each_row[3], each_row[4]))#for prediction_train,test.zip
                
        return txtlst
    
    def create_data(self, input_dir, file_names, output_dir, dtype, threadid):
        #input: formated folder; RAW_DATA + '/train/formatted/'
        #file_names: each txt file name
        #output_dir: DATA_DIR = '../../resources/data/' + DATASET#Ben: APOL
        name_lst = []
        i = 0
        sz = len(file_names)
        for f in file_names:
            print("Importing data {}/{} for {} in thread {}...".format(i, sz, dtype, threadid))
            i += 1
            # print(f)
            dset_id = f
            # print()
            loc = os.path.join(input_dir,dset_id+'.txt')#from 'formated folder'; i.e. formated txt file
            out = os.path.join(input_dir,dset_id+'.npy')
            import_data(loc, None, out)
            name_lst.append(out)
        # merge_n_split(name_lst, output_dir)
        merge(name_lst, output_dir, dtype, threadid)
        print('"merge" is finished!')




            # val_loc = RAW_DATA + '/val/data/'
            # output_dir = RAW_DATA + '/val/formatted/'
            # files = [f for f in os.listdir(val_loc) if '.txt' in f]
            # val_lst = carla_to_formatted(val_loc, files, output_dir, "val")
            # create_data(output_dir, val_lst, args.dir, "val", threadid)

            # test_loc = RAW_DATA + '/test_obs/data/'
            # output_dir = RAW_DATA + '/test_obs/formatted/'
            # files = [f for f in os.listdir(test_loc) if '.txt' in f]
            # test_lst = carla_to_formatted(test_loc, files, output_dir, "test")
            # create_data(output_dir, test_lst, args.dir, "test", threadid)