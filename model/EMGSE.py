import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
# import sru

spec_dim = 257
emg_dim = 32
total_dim = spec_dim + emg_dim
win_size = 64
device = torch.device('cuda:0')


class Dense_L(nn.Module):

    def __init__(self, in_size, out_size,bias=True):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_size, out_size, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.dense(x)
        return out

class Blstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self,x):

        out,_=self.blstm(x)
        out = out[:,:,:int(out.size(-1)/2)]+out[:,:,int(out.size(-1)/2):]  #merge_mode = 'sum'
        return out

class EMGSE_audio(nn.Module):
    
    def __init__(self,):
        super().__init__()
        self.spec_enc = nn.Sequential(
            Dense_L(257, 128, bias=True),
            Dense_L(128, 100, bias=True),
        )
        self.fuse = Dense_L(100,200,bias=True)
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=200, hidden_size=300, num_layers=2),
            nn.Linear(300, 257, bias=True),
            nn.ReLU(),
        )
    def forward(self,spec,emg):
        spec = self.spec_enc(spec)
        spec = self.fuse(spec)
        x = self.lstm_enc(spec)  
        return x#,spec,emg,spec

class EMGSE_all(nn.Module):
    
    def __init__(self,):
        super().__init__()
        self.spec_enc = nn.Sequential(
            Dense_L(257, 128, bias=True),
            Dense_L(128, 100, bias=True),
        )
        self.emg_enc = nn.Sequential(
            Dense_L(35*5*31, 200, bias=True),
            nn.Dropout(0.5),
            Dense_L(200, 100, bias=True),
            nn.Dropout(0.5)
        )
        self.fuse = Dense_L(200,200,bias=True)
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=200, hidden_size=300, num_layers=2),
            nn.Linear(300, 257, bias=True),
            nn.ReLU(),
        )
    def forward(self,spec,emg):
        emg = self.emg_enc(emg)
        spec = self.spec_enc(spec)
        x = torch.cat((spec,emg),2)
        f = self.fuse(x)
        out = self.lstm_enc(f)
        return out#,f#,emg,spec

class EMGSE_cheek(nn.Module):
    
    def __init__(self,):
        super().__init__()
        self.spec_enc = nn.Sequential(
            Dense_L(257, 128, bias=True),
            Dense_L(128, 100, bias=True),
        )
        self.emg_enc = nn.Sequential(
            Dense_L(28*5*31, 200, bias=True),
            nn.Dropout(0.5),
            Dense_L(200, 100, bias=True),
            nn.Dropout(0.5)
        )
        self.fuse = Dense_L(200,200,bias=True)
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=200, hidden_size=300, num_layers=2),
            nn.Linear(300, 257, bias=True),
            nn.ReLU(),
        )
    def forward(self,spec,emg):
        emg = self.emg_enc(emg)
        spec = self.spec_enc(spec)
        x = torch.cat((spec,emg),2)
        f = self.fuse(x)
        out = self.lstm_enc(f)
        return out#,f,emg,spec

