import torch

device = torch.device("cuda:1")
torch.manual_seed(1)

train = True
inference = True
"""
*** if train = False, must provide path to load model from and adjust hparams
below to those of the saved model (.pt extension). Uncomment and fill in below:
model_save_path =
model_save_path_LOSS = model_save_path+'_BEST_LOSS.pt'
model_save_path_ERROR = model_save_path+'_BEST_ERROR.pt'
"""

# temporary args:
batch_size = 10
maxiter = 50 # usually 20-30
num_threshold = 10 # usually 10 - num times valid loss can get worse before stopping

# model hyperparameters
lr = 1e-4
num_cnns = 5 # need to adjust code in models.py if you change this
nhu = 128 # LSTM size - 2048 is good (128 good for debugging)
nhu2 = 256 # Subsequent linear layer size - 1024 is good (256 good for debugging)
num_layers = 1

# model architecture, optimiser
model_arch = "cnn_lstm" # see options in archs_dict in models.py
opt = "adam" # adam or sgd

# file info
save_dir = "example" # ./save_dir/experiment_dir/experiment_file
hdf_file = "vctk_orig_MFB.hdf5" # name of HDF5 file, assumed ../data/hdf_file
extra = "extra info" # used in summary.csv

# extras that didn't improve results
dropout = 0
initweights = False # kaiming initialisation

# class labels
### labels of classes used in my dissertation - adapt to your dataset
orig_labels = ['ENG-W', 'SCT-IRE', 'NA']
broad_labels = ['ENG-W', 'SCT-IRE', 'NA', 'OCE', 'IND', 'SA']
narrow_labels = ['EngN', 'EngM', 'EngS', 'IreU', 'Ire', 'SctH', 'SctL', 'USN', 'USE',
                 'USS', 'USW-C', 'USNY', 'Aus', 'Ind', 'SA']
