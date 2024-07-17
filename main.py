import os
import sys
# os.chdir('../')
#sys.path.append('/raid/home/yoyowu/DeepPurpose')  
#print(os.getcwd())
#from DeepPurpose import DTI as models
import models
from utils import *
from dataset import *
import wandb

X_drugs, X_targets, y = read_file_training_dataset_drug_target_pairs('./toy_data/dti.txt')
print('Drug 1: ' + X_drugs[0])
print('Target 1: ' + X_targets[0])
print('Score 1: ' + str(y[0]))
X_drugs, X_targets, y = load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30)
print('Drug 1: ' + X_drugs[0])
print('Target 1: ' + X_targets[0])
print('Score 1: ' + str(y[0]))


fold=3
drug_encoding = 'CNN'
target_encoding = 'CNN'
train, val, test = data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)
train.head(1)

# use the parameters setting provided in the paper: https://arxiv.org/abs/1801.10193
config =generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 100, 
                         LR = 0.001, 
                         batch_size = 256,
                         cnn_drug_filters = [32,64,96],
                         cnn_target_filters = [32,64,96],
                         cnn_drug_kernels = [4,6,8],
                         cnn_target_kernels = [4,8,12],
                         model_save_path = "/raid/home/yoyowu/DeepPurpose/checkpoint/0111DTI_base_fold{}.txt".format(fold)
                        )

wandb.init(project="DeepPurpose", name="base_DTI_{}.txt".format(fold), config=config)
model = models.model_initialize(**config)
model.train(train, val, test)
#model.save_model("/raid/home/yoyowu/DeepPurpose/checkpoint")