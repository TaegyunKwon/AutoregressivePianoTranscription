import torch as th
from torch import nn
import numpy as np
from torch.nn import functional as F
import nnAudio
from torchaudio import transforms
from natten import NeighborhoodAttention2D

# from .cqt import CQT
from .constants import SR, HOP
from .context import random_modification, update_context
# from .cqt import MultiCQT
from .midispectrogram import CombinedSpec, MidiSpec
from .model import MIDIFrontEnd, FilmLayer, HarmonicDilatedConv

class TransModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.local_model_name = config.local_model_name
        self.lm_model_name = config.lm_model_name
        self.n_fft = config.n_fft
        self.hidden_per_pitch = config.hidden_per_pitch
        self.pitchwise = config.pitchwise_lstm

        self.trans_model = NATTEN(config.hidden_per_pitch)
        self.embedding = nn.Embedding(5, 4)
        self.output = nn.Linear(24, 5)

    def forward(self, features, condition, mask):
        # Features: B x T x H x 88
        # condition: B x T x 88
        # mask: B x T x 88
        cond_emb = self.embedding(condition)
        cat = th.cat((features.permute(0,1,3,2), cond_emb, mask.unsqueeze(-1)), dim=-1)
        x = self.trans_model(cat)
        out = self.output(x)
        return out

def get_conv2d_block(channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1],
                        use_film=False, n_f=None):
    modules = [
        nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
        nn.ReLU()
        ]
    if use_film:
        modules.append(FilmLayer(n_f, channel_out))
    if pool_size != None:
        modules.append(nn.MaxPool2d(pool_size))
    modules.append(nn.InstanceNorm2d(channel_out))
    return nn.Sequential(*modules)


class HPP_FC(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit, hidden_per_pitch, use_film,
                 n_per_pitch=5, frontend_kernel_size=7):
        super().__init__()
        # input is batch_size * 3 channel * frames * n_mels
        
        self.n_h = hidden_per_pitch // 2
        self.conv_front = nn.Sequential(
            get_conv2d_block(3, cnn_unit, kernel_size=frontend_kernel_size, use_film=use_film, n_f=n_mels),
            get_conv2d_block(cnn_unit, cnn_unit, kernel_size=frontend_kernel_size, use_film=use_film, n_f=n_mels),
            get_conv2d_block(cnn_unit, cnn_unit, kernel_size=frontend_kernel_size, use_film=use_film, n_f=n_mels))
        self.hdc0 = nn.Sequential(
            HarmonicDilatedConv(cnn_unit, self.n_h, n_per_pitch),
             HarmonicDilatedConv(self.n_h, self.n_h, n_per_pitch),
             HarmonicDilatedConv(self.n_h, self.n_h, n_per_pitch),
            get_conv2d_block(self.n_h, self.n_h, pool_size=[1, n_per_pitch], 
                                  dilation=[1, 12*n_per_pitch]),
            )
        self.hdc1 = nn.Sequential(
            get_conv2d_block(self.n_h, self.n_h, dilation=[1, 12]),
            get_conv2d_block(self.n_h, self.n_h, [5,1]),
            get_conv2d_block(self.n_h, self.n_h, [5,1]),
            get_conv2d_block(self.n_h, self.n_h, [5,1])
            )
        
        self.conv = get_conv2d_block(self.n_h, 8, [1,1])
        self.fc = nn.Sequential(
            nn.Linear(8*99, fc_unit),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(fc_unit, self.n_h*88),
            nn.LayerNorm(self.n_h*88),
            nn.ReLU()
            )
        
    def forward(self, x):
        # x: B x 3 x n_mels x T 
        batch_size = x.shape[0]
        T = x.shape[3]
        x = self.conv_front(x.permute(0, 1, 3, 2))
        x = self.hdc0(x)  # B, 128, T, 99
        x_hdc = self.hdc1(x[:,:,:,:88])  # B, 128, T, 88
        x_conv = self.conv(x)  # B, 8, T, 99
        x_fc = self.fc(x_conv.permute(0,2,1,3).flatten(-2)).reshape(batch_size, T, self.n_h, 88)
        x_fc = x_fc.permute(0, 2, 1, 3) # B, H/2, T, 88
        x = th.cat((x_hdc, x_fc), dim=1)  # B, H, T, 88
        return x.permute(0, 2, 3, 1)  # B, T, 88, H
    

class NATTEN(nn.Module):
    def __init__(self, hidden_per_pitch, window=25, n_unit=24, n_head=4, n_layers=2):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers

        self.linear = nn.Sequential(nn.Linear(hidden_per_pitch+5, n_unit),
                                    nn.ReLU())
        self.na = nn.Sequential(*([NeighborhoodAttention2D(n_unit, 4, window)]* n_layers))


    def forward(self, x):
        # x: B x T x 88 x H+5
        cat = self.linear(x)
        na_out = self.na(cat) # B x T x 88 x N
        return na_out
        

        

    