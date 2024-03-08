import torch as th
from torch import nn
import numpy as np
from torch.nn import functional as F
import nnAudio
from torchaudio import transforms
from natten import NeighborhoodAttention2D, NeighborhoodAttention1D

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
        self.cnn_unit = config.cnn_unit
        self.hidden_per_pitch = config.hidden_per_pitch
        self.pitchwise = config.pitchwise_lstm

        # self.trans_model = NATTEN(config.hidden_per_pitch)
        self.frontend = MIDIFrontEnd(n_per_pitch=config.n_per_pitch)
        self.front_block = nn.Sequential(
            ConvFilmBlock(3, config.cnn_unit, 3, 1, use_film=True, n_f=495),
            ConvFilmBlock(config.cnn_unit, config.cnn_unit, 3, 1, use_film=True, n_f=495),
            ConvFilmBlock(config.cnn_unit, config.cnn_unit, 3, 1, pool_size=(1,5), use_film=True, n_f=495)
            )
        self.front_block_vel = nn.Sequential(
            ConvFilmBlock(3, config.cnn_unit, 3, 1, use_film=True, n_f=495),
            ConvFilmBlock(config.cnn_unit, config.cnn_unit, 3, 1, use_film=True, n_f=495),
            ConvFilmBlock(config.cnn_unit, config.cnn_unit, 3, 1, pool_size=(1,5), use_film=True, n_f=495)
            )
        self.condition_block = ConditionBlock(config.n_unit)
        self.middle_block = MiddleBlock(config.cnn_unit + config.n_unit, config.cnn_unit)
        self.middle_block_vel = MiddleBlock(config.cnn_unit + config.n_unit, config.cnn_unit)
        self.high_block = HighBlock(config.cnn_unit + config.n_unit, config.n_unit)
        self.high_block_vel = HighBlock(config.cnn_unit + config.n_unit, config.n_unit)
        self.output = nn.Linear(config.n_unit, 5)
        self.output_vel = nn.Linear(config.n_unit, 128)

    def forward(self, audio, condition, vel_condition, mask):
        # condition: B x T x 88
        # mask: B x T x 88

        spec = self.frontend(audio)  # B, 3, F, T
        x = self.front_block(spec.permute(0,1,3,2))  # B, C, T, 99
        c = self.condition_block(condition, vel_condition, mask)  # B N T 88
        c_pad = F.pad(c, (0, 11))  # B, N, 99, T
        cat_level_1 = th.cat((x, c_pad), dim=1)  # B, C+N, T, 99
        x = self.middle_block(cat_level_1)  # B, C, T, 88
        cat_level_2 = th.cat((x, c), dim=1)  # B, C+N, T, 88
        x = self.high_block(cat_level_2)  # B, T, 88, N
        out = self.output(x)  # B, T, 88, 5

        x_vel = self.front_block_vel(spec.permute(0,1,3,2))  # B, C, T, 99
        cat_level_1_vel = th.cat((x_vel, c_pad), dim=1)  # B, C+N, T, 99
        x_vel = self.middle_block_vel(cat_level_1_vel)  # B, C, T, 88
        cat_level_2_vel = th.cat((x_vel, c), dim=1)  # B, C+N, T, 88
        x_vel = self.high_block_vel(cat_level_2_vel)  # B, T, 88, N
        vel_out = self.output_vel(x_vel)  # B, T, 88, 128
        return out, vel_out


class MIDIFrontEnd(nn.Module):
    def __init__(self, n_per_pitch=3, detune=0.0) -> None:
        # Detune: semitone unit. 0.5 means 50 cents.
        super().__init__()
        self.midi_low = MidiSpec(1024, n_per_pitch)
        self.midi_mid = MidiSpec(4096, n_per_pitch)
        self.midi_high = MidiSpec(8192, n_per_pitch)

    def forward(self, audio, detune_list=None):
        midi_low = self.midi_low(audio, detune_list)
        midi_mid = self.midi_mid(audio, detune_list)
        midi_high = self.midi_high(audio, detune_list)
        spec = th.stack([midi_low, midi_mid, midi_high], dim=1)
        return spec # B, 3, F, T


class ConvFilmBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, dilation, pool_size=None, use_film=True, n_f=88):
        super().__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation)
        self.relu = nn.ReLU()
        if use_film:
            self.film = FilmLayer(n_f, channel_out)
        if pool_size != None:
            self.pool = nn.MaxPool2d(pool_size)
        self.norm = nn.InstanceNorm2d(channel_out)
        self.pool_size = pool_size
        self.use_film = use_film

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.use_film:
            x = self.film(x)
        if self.pool_size != None:
            x = self.pool(x)
        x = self.norm(x)
        return x


class ConditionBlock(nn.Module):
    def __init__(self, n_unit):
        super().__init__()
        self.n_unit = n_unit
        self.emb_layer_y = nn.Embedding(5, 4)
        self.emb_layer_v = nn.Embedding(128, 2)
        self.lstm = nn.LSTM(7, n_unit//2, 2, batch_first=True, bidirectional=True)
        self.na1_t = NeighborhoodAttention1D(n_unit, 1, 7)
        self.na1_f = NeighborhoodAttention1D(n_unit, 1, 87)
        self.na2_t = NeighborhoodAttention1D(n_unit, 1, 7)
        self.na2_f = NeighborhoodAttention1D(n_unit, 1, 87)

    def forward(self, y, v, m):
        # y: B x T x 88. Label
        # v: B x T x 88. Velocity
        # m: B x T x 88. Mask
        B = y.shape[0]
        T = y.shape[1]
        y_emb = self.emb_layer_y(y)
        v_emb = self.emb_layer_v(v)
        cat = th.cat((y_emb, v_emb, m.unsqueeze(-1)), dim=-1)  
        cat_pitchwise = cat.permute(0, 2, 1, 3).reshape(B*88, T, 7)
        x, _ = self.lstm(cat_pitchwise) # B*88, T, N
        x = self.na1_t(x)
        x_timewise = x.reshape(B, 88, T, self.n_unit).permute(0,2,1,3).reshape(B*T, 88, self.n_unit)
        x = self.na1_f(x_timewise)
        x_pitchwise = x.reshape(B, T, 88, self.n_unit).permute(0,2,1,3).reshape(B*88, T, self.n_unit)
        x = self.na2_t(x_pitchwise)
        x_timewise = x.reshape(B, 88, T, self.n_unit).permute(0,2,1,3).reshape(B*T, 88, self.n_unit)
        x = self.na2_f(x_timewise)
        return x.reshape(B, T, 88, self.n_unit).permute(0, 3, 1, 2) # B, N, T, 88


class HarmonicDilatedConv(nn.Module):
    def __init__(self, c_in, c_out, n_per_pitch=4, use_film=False, n_f=None) -> None:
        super().__init__()
        dilations = [round(12*np.log2(a)*n_per_pitch) for a in range(2, 10)]
        self.conv = nn.ModuleDict()
        for i, d in enumerate(dilations):
            self.conv[str(i)] = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, d])
        self.use_film = use_film
        if use_film:
            self.film = FilmLayer(n_f, c_out)
    def forward(self, x):
        x = self.conv['0'](x) + self.conv['1'](x) + self.conv['2'](x) + self.conv['3'](x) + \
            self.conv['4'](x) + self.conv['5'](x) + self.conv['6'](x) + self.conv['7'](x)
        if self.use_film:
            x = self.film(x)
        x = th.relu(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_input, cnn_unit):
        super().__init__()
        self.hdc0 = HarmonicDilatedConv(n_input, cnn_unit, 1)
        self.hdc1 = HarmonicDilatedConv(cnn_unit, cnn_unit, 1)
        self.hdc2 = HarmonicDilatedConv(cnn_unit, cnn_unit, 1)
        self.block = nn.Sequential(
            ConvFilmBlock(cnn_unit, cnn_unit, [3,1], 1, use_film=True, n_f=88),
            ConvFilmBlock(cnn_unit, cnn_unit, [3,1], 1, use_film=True, n_f=88),
        )
    
    def forward(self, x):
        # x: B x N+C x T x 99
        x = self.hdc0(x)
        x = self.hdc1(x)
        x = self.hdc2(x)[:,:,:,:88]
        x = self.block(x)
        return x  # B C T 88


class HighBlock(nn.Module):
    def __init__(self, input_unit, n_unit):
        super().__init__()
        self.n_unit = n_unit
        self.lstm = nn.LSTM(input_unit, n_unit//2, 2, batch_first=True, bidirectional=True)
        self.na_1t = NeighborhoodAttention1D(n_unit, 1, 7)
        self.na_1f = NeighborhoodAttention1D(n_unit, 1, 87)

    def forward(self, x):
        #  x: B x C+N x T x 88
        B = x.shape[0]
        H = x.shape[1]
        T = x.shape[2]
        x = x.permute(0, 3, 2, 1).reshape(B*88, T, H)
        x, c = self.lstm(x) 
        x = self.na_1t(x)
        x_timewise = x.reshape(B, 88, T, self.n_unit).permute(0,2,1,3).reshape(B*T, 88, self.n_unit)
        x = self.na_1f(x_timewise)
        return x.reshape(B, T, 88, self.n_unit)


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
  
        
class LSTM_NATTEN(nn.Module):
    def __init__(self, hidden_per_pitch, window=25, n_unit=24, n_head=4, n_layers=2):
        super().__init__()
        self.n_head = n_head
        self.n_layers = n_layers

        self.lstm = nn.LSTM(hidden_per_pitch+5, n_unit//2, 2, batch_first=True, bidirectional=True)
        self.na = nn.Sequential(*([NeighborhoodAttention2D(n_unit, 4, window)]* n_layers))


    def forward(self, x):
        B = x.shape[0]
        H = x.shape[-1]
        T = x.shape[1]
        # x: B x T x 88 x H+5
        x = x.permute(0, 2, 1, 3).reshape(B*88, T, H)
        x, c = self.lstm(x)  
        x = x.reshape(B, 88, T, -1).permute(0,2,1,3)
        na_out = self.na(x) # B x T x 88 x N
        return na_out

        

    