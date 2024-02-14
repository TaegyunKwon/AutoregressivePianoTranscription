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

class InFillModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.local_model_name = config.local_model_name
        self.lm_model_name = config.lm_model_name
        self.n_fft = config.n_fft
        self.hidden_per_pitch = config.hidden_per_pitch
        self.pitchwise = config.pitchwise_lstm
        self.frontend = MIDIFrontEnd(config.n_per_pitch)
        if self.local_model_name == 'HPP_FC':
            self.local_model = HPP_FC(config.n_mels, config.cnn_unit, config.fc_unit,
                                      config.hidden_per_pitch, config.use_film, config.n_per_pitch, config.frontend_filter_size)
            self.local_model_vel = HPP_FC(config.n_mels, config.cnn_unit, config.fc_unit,
                                      config.hidden_per_pitch, config.use_film, config.n_per_pitch, config.frontend_filter_size)
        if self.lm_model_name == 'NATTEN':
            self.lm_model = NATTEN(config.hidden_per_pitch)
            self.lm_model_vel = NATTEN(config.hidden_per_pitch)
        else:
            raise KeyError(f'wrong model:{self.local_model_name}')
        self.output = nn.Linear(config.hidden_per_pitch, 5)
        self.output_vel = nn.Linear(config.hidden_per_pitch, 128)

    def forward(self, audio, condition, mask, return_softmax=False):
        frontend = self.frontend(audio)
        mid_features = self.local_model(frontend)
        mid_features_vel = self.local_model_vel(frontend)
        features = self.lm_model(mid_features, condition, mask)
        features_vel =  self.lm_model_vel(mid_features_vel, condition, mask)
        out = self.output(features)
        out_vel = self.output_vel(features_vel)
        if return_softmax:
            out = F.log_softmax(out, dim=-1)
            out_vel = F.log_softmax(out_vel, dim=-1)
        return out, out_vel

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
    def __init__(self, hidden_per_pitch, window=25, n_head=4, n_layers=2):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers
        self.hidden_per_pitch = hidden_per_pitch

        self.linear = nn.Sequential(nn.Linear(hidden_per_pitch+3, hidden_per_pitch),
                                    nn.ReLU())
        self.na = nn.Sequential(*([NeighborhoodAttention2D(hidden_per_pitch, 4, window)]* n_layers))


    def forward(self, x, condition, mask):
        # x: B x T x H x 88
        # condition: B x T x 88 x 2 (5+vel)
        # mask: B x T x 88
        cat = th.cat((x, condition, mask.unsqueeze(-1)), dim=-1) # B x T x 88 x H+3
        cat = self.linear(cat)
        na_out = self.na(cat) # B x T x 88 x H
        return na_out
        

        

    