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

class ConditionEmbedding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.emb_layer_r = nn.Embedding(5, out_dim//4)
        self.emb_layer_p = nn.Embedding(5, out_dim//4)
        self.emb_layer_r_vel = nn.Embedding(128, out_dim//4)
        self.emb_layer_p_vel = nn.Embedding(128, out_dim//4)

    def forward(self, r, p, r_vel, p_vel):
        # shape: B, T, 88
        x1 = self.emb_layer_r(r)
        x2 = self.emb_layer_p(p)
        x3 = self.emb_layer_r_vel(r_vel)
        x4 = self.emb_layer_p_vel(p_vel)
        return th.cat((x1, x2, x3, x4), dim=-1) # B, T, 88, out_dim


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
        c3_out = 128
        self.emb1 = ConditionEmbedding(config.cnn_unit)
        self.conv_3 = HarmonicDilatedConv(config.cnn_unit, c3_out, 1)
        self.emb2 = ConditionEmbedding(c3_out)
        self.conv_4 = HarmonicDilatedConv(c3_out, c3_out, 1)
        self.emb3 = ConditionEmbedding(c3_out)
        self.conv_5 = HarmonicDilatedConv(c3_out, c3_out, 1)
        self.emb4 = ConditionEmbedding(c3_out)

        self.block_4 = ConvFilmBlock(c3_out, c3_out, [3,1], dilation=[1, 12], n_f=99)
        self.block_5 = ConvFilmBlock(c3_out, c3_out, [3,1], dilation=[1, 12])
        self.block_6 = ConvFilmBlock(c3_out, c3_out, [5,1], 1)
        self.block_7 = ConvFilmBlock(c3_out, c3_out, [5,1], 1)
        self.block_8 = ConvFilmBlock(c3_out, c3_out, [5,1], 1)

        self.emb5 = ConditionEmbedding(c3_out)
        self.lstm = nn.LSTM(c3_out, config.hidden_per_pitch//2, 2, batch_first=True, bidirectional=True)

        self.output = nn.Linear(config.hidden_per_pitch, 5)

    def forward(self, audio, r, p, r_vel, p_vel):
        # condition: B x T x 88
        # mask: B x T x 88

        spec = self.frontend(audio)  # B, 3, F, T
        B = spec.shape[0]
        T = spec.shape[3]
        x = self.front_block(spec.permute(0,1,3,2))  # B, C, T, 99
        c = self.emb1(r, p, r_vel, p_vel).permute(0,3,1,2)  # B, C, T, 88
        x = x + F.pad(c, (0,11))
        x = self.conv_3(x)
        c = self.emb2(r, p, r_vel, p_vel).permute(0,3,1,2)  # B, C, T, 88
        x = x + F.pad(c, (0,11))
        x = self.conv_4(x)
        c = self.emb3(r, p, r_vel, p_vel).permute(0,3,1,2)  # B, C, T, 88
        x = x+ F.pad(c, (0,11))
        x = self.conv_5(x)
        c = self.emb4(r, p, r_vel, p_vel).permute(0,3,1,2)  # B, C, T, 88
        x = x+ F.pad(c, (0,11))
        x = self.block_4(x)
        x = x[:,:,:,:88]

        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x 
        c = self.emb5(r, p, r_vel, p_vel).permute(0,3,1,2)  # B, C, T, 88
        x = x + c

        x = x.permute(0, 3, 2, 1).reshape(B*88, T, 128)
        x, _ = self.lstm(x)
        x = x.reshape(B, 88, T, self.hidden_per_pitch).permute(0,2,1,3)
        x = self.output(x) # B, T, 88, 5

        return x


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

        

    