# Imports
import torch.nn as nn  # nn.Module
from hparams import *

"""6 different models used during experiments. CNN_LSTM had best results."""

# CNN_LSTM
class CNN_LSTM(nn.Module):
    def __init__(self, idim, num_classes, nhu, nhu2, nlay, dropout=dropout):
        super(CNN_LSTM, self).__init__()
        self.nhu = nhu  # number of hidden units
        self.nhu2 = nhu2 # bigger number of hidden units for later linear
        self.nlay = nlay  # number of hidden layers
        self.nhu3 = 60

        # Architecture: [CNN(+ReLU)]*num_cnns -> LSTM -> mean -> Linear -> ReLU -> Linear
        self.conv = nn.Conv1d(idim, self.nhu3, 3, stride=1, padding=1) # nin=40, nout=60
        self.conv2 = nn.Conv1d(self.nhu3, self.nhu3, 3, stride=1, padding=1) # nin=60, nout=60
        self.relu_conv = nn.ReLU()
        self.lstm = nn.LSTM(input_size=self.nhu3, hidden_size=nhu, num_layers=nlay, batch_first=True)#, dropout=dropout)
        self.lin_1 = nn.Sequential(nn.Linear(nhu, nhu2), nn.ReLU(), nn.Dropout(dropout))
        self.final_linear = nn.Linear(nhu2, num_classes)

        if initweights == True:
            self.apply(init_weights)

    def forward(self, batch): # [batch_size, seq_len, frame_dim (40)]
        # Set initial hidden and cell states
        h0 = torch.zeros(self.nlay, batch.size(0), self.nhu).to(device)
        c0 = torch.zeros(self.nlay, batch.size(0), self.nhu).to(device)

        # CNNs
        batch2 = batch.transpose(2,1) # flip seq_len and frame_dim
        # CNN 1
        out_conv=self.conv(batch2) # needs batch, frame_dim, seq_len
        out_relu_conv=self.relu_conv(out_conv)
        # CNN 2
        out_conv2=self.conv2(out_relu_conv)
        out_relu_conv2=self.relu_conv(out_conv2)
        # CNN 3
        out_conv3=self.conv2(out_relu_conv2)
        out_relu_conv3=self.relu_conv(out_conv3)
        # CNN 4
        out_conv4=self.conv2(out_relu_conv3)
        out_relu_conv4=self.relu_conv(out_conv4)
        # CNN 5
        out_conv5=self.conv2(out_relu_conv4)
        out_relu_conv5=self.relu_conv(out_conv5)

        # Forward propagate LSTM
        out_relu_conv_final = out_relu_conv5.transpose(1,2) # flip to get batch, seq_len, frame_dim for lstm
        out,_=self.lstm(out_relu_conv_final,(h0,c0)) # out: tensor of shape (batch_size, seq_length, hidden_size)

        out_mean = torch.mean(out, 1) # out before mean is [num_files, seq_len, nhu], avg over seq_len [num_files, nhu]
        out_relu = self.lin_1(out_mean)

        # Decode the hidden state of the last time step
        final_out = self.final_linear(out_relu) # in: [num_files, nhu2], out: [num_files, num_classes]
        return final_out


# CNN-BLSTM
class CNN_BLSTM(nn.Module):
    def __init__(self, idim, num_classes, nhu, nhu2, nlay, dropout=dropout):
        super(CNN_BLSTM, self).__init__()
        self.nhu = nhu  # number of hidden units
        self.nhu2 = nhu2 # bigger number of hidden units for later linear
        self.nlay = nlay  # number of hidden layers
        self.nhu3 = 60 # dims of CNN
        # Architecture: [CNN(+ReLU)]*num_cnns -> BLSTM -> mean -> Linear -> ReLU -> Linear
        self.conv = nn.Conv1d(idim, self.nhu3, 3, stride=1, padding=1) # nin=40, nout=60
        self.conv2 = nn.Conv1d(self.nhu3, self.nhu3, 3, stride=1, padding=1) # nin=60, nout=60
        self.relu_conv = nn.ReLU()
        self.lstm = nn.LSTM(input_size=self.nhu3, hidden_size=nhu, num_layers=nlay, dropout=dropout, bidirectional=True, batch_first=True)
        self.lin_1 = nn.Sequential(nn.Linear(nhu*2, nhu2), nn.ReLU(), nn.Dropout(dropout))
        self.final_linear = nn.Linear(nhu2, num_classes)

    def forward(self, batch): # [batch_size, seq_len, frame_dim (40)]
        # Set initial hidden and cell states
        h0 = torch.zeros(self.nlay*2, batch.size(0), self.nhu).to(device)
        c0 = torch.zeros(self.nlay*2, batch.size(0), self.nhu).to(device)

        # CNNs
        batch2 = batch.transpose(2,1) # flip seq_len and frame_dim
        # CNN 1
        out_conv=self.conv(batch2) # needs batch, frame_dim, seq_len
        out_relu_conv=self.relu_conv(out_conv)
        # CNN 2
        out_conv2=self.conv2(out_relu_conv)
        out_relu_conv2=self.relu_conv(out_conv2)
        # CNN 3
        out_conv3=self.conv2(out_relu_conv2)
        out_relu_conv3=self.relu_conv(out_conv3)
        # CNN 4
        out_conv4=self.conv2(out_relu_conv3)
        out_relu_conv4=self.relu_conv(out_conv4)
        # CNN 5
        out_conv5=self.conv2(out_relu_conv4)
        out_relu_conv5=self.relu_conv(out_conv5)

        # Forward propagate LSTM
        out_relu_conv_final = out_relu_conv5.transpose(1,2) # flip to get batch, seq_len, frame_dim for lstm
        out,_=self.lstm(out_relu_conv_final,(h0,c0)) # out: tensor of shape (batch_size, seq_length, hidden_size)

        out_mean = torch.mean(out, 1) # out before mean is [num_files, seq_len, nhu], avg over seq_len [num_files, nhu]
        out_relu = self.lin_1(out_mean)

        # Decode the hidden state of the last time step
        final_out = self.final_linear(out_relu) # in: [num_files, nhu2], out: [num_files, num_classes]
        return final_out


# LSTM
class LSTM_RELU(nn.Module):
    #def __init__(self, input_size, hidden_size, num_layers, num_classes):
    def __init__(self, idim, num_classes, nhu, nhu2, nlay, dropout=dropout):
        super(LSTM_RELU, self).__init__()
        self.nhu = nhu  # number of hidden units
        self.nhu2 = nhu2 # bigger number of hidden units for later linear
        self.nlay = nlay  # number of hidden layers

        # Architecture: LSTM -> mean -> Linear -> ReLU -> Linear
        self.lstm = nn.LSTM(input_size=idim, hidden_size=nhu, num_layers=nlay, batch_first=True)#, dropout=dropout)
        self.lin_1 = nn.Sequential(nn.Linear(nhu, nhu2), nn.ReLU(), nn.Dropout(dropout))
        self.final_linear = nn.Linear(nhu2, num_classes)

        if initweights == True:
            self.apply(init_weights)

    def forward(self, batch):
        #print('batch', batch.size())
        # Set initial hidden and cell states - not necessary for LSTM, initialised with these
        h0 = torch.zeros(self.nlay, batch.size(0), self.nhu).to(device)
        c0 = torch.zeros(self.nlay, batch.size(0), self.nhu).to(device)
        #print('h0', h0.size(), 'c0', c0.size())

        # Forward propagate LSTM
        out, _ = self.lstm(batch, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out_mean = torch.mean(out, 1) # out before mean is [num_files, seq_len, nhu], avg over seq_len [num_files, nhu]
        out_relu = self.lin_1(out_mean)

        # Decode the hidden state of the last time step
        final_out = self.final_linear(out_relu) # in: [num_files, nhu2], out: [num_files, num_classes]
        return final_out


# LSTM
class BLSTM_RELU(nn.Module):
    #def __init__(self, input_size, hidden_size, num_layers, num_classes):
    def __init__(self, idim, num_classes, nhu, nhu2, nlay, dropout=dropout):
        super(BLSTM_RELU, self).__init__()
        self.nhu = nhu  # number of hidden units
        self.nhu2 = nhu2 # bigger number of hidden units for later linear
        self.nlay = nlay  # number of hidden layers

        # Architecture: BLSTM -> mean -> Linear -> ReLU -> Linear
        self.lstm = nn.LSTM(input_size=idim, hidden_size=nhu, num_layers=nlay, dropout=dropout, bidirectional=True, batch_first=True)
        self.lin_1 = nn.Sequential(nn.Linear(nhu*2, nhu2), nn.ReLU(), nn.Dropout(dropout))
        self.final_linear = nn.Linear(nhu2, num_classes)

        if initweights == True:
            self.apply(init_weights)

    def forward(self, batch):
        # Set initial hidden and cell states - not necessary for LSTM, initialised with these
        h0 = torch.zeros(self.nlay*2, batch.size(0), self.nhu).to(device) # batch.size = (3, seq_len, feat_dime)
        c0 = torch.zeros(self.nlay*2, batch.size(0), self.nhu).to(device)
        #print('h0,c0', h0.size(), c0.size())

        # Forward propagate LSTM
        out, _ = self.lstm(batch, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #print('lstm out', out.size())
        out_mean = torch.mean(out, 1) # out before mean is [num_files, seq_len, nhu], avg over seq_len [num_files, nhu]
        #print('mean out', out_mean.size())
        out_relu = self.lin_1(out_mean)
        #print('relu out', out_relu.size())

        # Decode the hidden state of the last time step
        final_out = self.final_linear(out_relu) # in: [num_files, nhu2], out: [num_files, num_classes]
        #print('final out', final_out.size())

        return final_out


class LSTM(nn.Module):
    def __init__(self, idim, num_classes, nhu, nlay):
        super(LSTM, self).__init__()
        # Architecture: LSTM -> mean -> Linear
        self.nhu = nhu  # number of hidden units
        self.nhu2 = nhu2 # bigger number of hidden units for later linear
        self.nlay = nlay  # number of hidden layers
        self.lstm = nn.LSTM(input_size=idim, hidden_size=nhu, num_layers=nlay, batch_first=True)#, dropout=dropout)
        self.fc = nn.Linear(nhu, num_classes)

        if initweights == True:
            self.apply(init_weights)

    def forward(self, batch):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.nlay, batch.size(0), self.nhu).to(device)
        c0 = torch.zeros(self.nlay, batch.size(0), self.nhu).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(batch, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out_mean = torch.mean(out, 1) # out before mean is [num_files, seq_len, nhu], avg over seq_len [num_files, nhu]

        # Decode the hidden state of the last time step
        final_out = self.fc(out_mean) # input [num_files, nhu], output should be num_classes (odim)

        return final_out


# BLSTM
class BLSTM(nn.Module):
    #def __init__(self, input_size, hidden_size, num_layers, num_classes):
    def __init__(self, idim, num_classes, nhu, nlay, dropout=dropout):
        super(BLSTM, self).__init__()
        self.nhu = nhu  # number of hidden units
        self.nlay = nlay  # number of hidden layers
        self.lstm = nn.LSTM(input_size=idim, hidden_size=nhu, num_layers=nlay, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(nhu*2, num_classes) # idim 40 for now, change to batch-dependent sequence length feat_dim, num_classes
        #self.dnn = nn.Linear(nhu, odim)

    def forward(self, batch):
        # Set initial hidden and cell states - not necessary for LSTM, initialised with these
        h0 = torch.zeros(self.nlay*2, batch.size(0), self.nhu).to(device) # batch.size = (3, seq_len, feat_dime)
        c0 = torch.zeros(self.nlay*2, batch.size(0), self.nhu).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(batch, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # take output of lstm, use .mean (check dims) - instead of n frames of 40 dim, just one frame of 40 dims
        out_mean = torch.mean(out, 1) # out before mean is [num_files, seq_len, nhu], avg over seq_len [num_files, nhu]
        # Decode the hidden state of the last time step
        final_out = self.fc(out_mean) # input [num_files, nhu], output should be num_classes (odim)
        return final_out


archs_dict = {'lstm':LSTM,
        'blstm':BLSTM,
        'lstm_relu':LSTM_RELU,
        'blstm_relu':BLSTM_RELU,
        'cnn_lstm':CNN_LSTM,
        'cnn_blstm':CNN_BLSTM}
