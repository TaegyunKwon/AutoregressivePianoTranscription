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

def sinusoids(length, channels, max_timescale=500):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = th.exp(-log_timescale_increment * th.arange(channels // 2))
    scaled_time = th.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return th.cat([th.sin(scaled_time), th.cos(scaled_time)], dim=1)

class TransModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.local_model_name = config.local_model_name
        self.lm_model_name = config.lm_model_name
        self.n_fft = config.n_fft
        self.cnn_unit = config.cnn_unit
        self.hidden_per_pitch = config.hidden_per_pitch
        self.pitchwise = config.pitchwise_lstm

        self.frontend = MIDIFrontEnd(n_per_pitch=config.n_per_pitch)
        self.front_block = nn.Sequential(
            ConvFilmBlock(3, config.cnn_unit, 3, 1, use_film=True, n_f=495),
            ConvFilmBlock(config.cnn_unit, config.cnn_unit, 3, 1, use_film=True, n_f=495),
            ConvFilmBlock(config.cnn_unit, config.cnn_unit, 3, 1, pool_size=(1,5), use_film=True, n_f=495)
            )
        self.context_emb = ContextNet(bw=True, out_dim=config.cnn_unit)
        self.cross_attention = CrossAttention(config.cnn_unit)
        self.context_emb2 = ContextNet(bw=True, out_dim=config.cnn_unit)
        self.middle_block = MiddleBlock(config.cnn_unit, config.cnn_unit)
        self.context_emb3 = ContextNet(bw=False, out_dim=4)
        self.lstm = nn.LSTM(config.cnn_unit+4, config.lstm_unit, 2, batch_first=True, bidirectional=False)
        self.output = nn.Linear(config.lstm_unit, 5)

        self.front_block_v = nn.Sequential(
            ConvFilmBlock(3, config.cnn_unit, 3, 1, use_film=True, n_f=495),
            ConvFilmBlock(config.cnn_unit, config.cnn_unit, 3, 1, use_film=True, n_f=495),
            ConvFilmBlock(config.cnn_unit, config.cnn_unit, 3, 1, pool_size=(1,5), use_film=True, n_f=495)
            )
        self.context_emb_v = ContextNet(bw=True, out_dim=config.cnn_unit)
        self.cross_attention_v = CrossAttention(config.cnn_unit)
        self.context_emb2_v = ContextNet(bw=True, out_dim=config.cnn_unit)
        self.middle_block_v = MiddleBlock(config.cnn_unit, config.cnn_unit)
        self.context_emb3_v = ContextNet(bw=False, out_dim=4)
        self.middle_block_v = MiddleBlock(config.cnn_unit, config.cnn_unit)
        self.lstm_v = nn.LSTM(config.cnn_unit+4, config.lstm_unit, 2, batch_first=True, bidirectional=False)
        self.output_vel = nn.Linear(config.lstm_unit, 128)

    def forward(self, audio, c_mask, init_frames, init_vels, infer=False, c_target=None, target_pitch=None, 
                          init_state=None, init_onset_time=None, init_onset_vel=None,
                          init_h=None, init_h_vel=None, unpad_start=False, unpad_end=False):
        if not infer:
            # c_mask: cat(last_states, f, v ,b, b_v) : B, T, 88, 5
            x, x_vel = self.forward_mask(audio, c_mask) # B C T 88
            B = x.shape[0]
            T = x.shape[2]

            c_fw = self.context_emb3(c_target)  # B, T, 88, N
            x = th.cat((x, c_fw.permute(0,3,1,2)), dim=1)  # B, C+N, T, 88
            x = x.permute(0, 3, 2, 1).reshape(B*88, T, -1)
            x, _ = self.lstm(x)
            x = x.reshape(B, 88, T, -1).permute(0,2,1,3)
            output = self.output(x)  # B, T, 88, 5

            c_fw = self.context_emb3_v(c_target)  # B, T, 88, N
            x = th.cat((x_vel, c_fw.permute(0,3,1,2)), dim=1)  # B, C+N, T, 88
            x = x.permute(0, 3, 2, 1).reshape(B*88, T, -1)
            x, _ = self.lstm_v(x)
            x = x.reshape(B, 88, T, -1).permute(0,2,1,3)
            vel_out = self.output_vel(x)  # B, T, 88, 128

            return output, vel_out
        if infer:
            x, x_v = self.forward_mask(audio, c_mask, unpad_start, unpad_end) # B C T 88
            B = x.shape[0]
            T = x.shape[2]
            device = x.device

            
            frame = init_frames.to(device)
            vel = init_vels.to(device)

            if init_state == None:
                init_state = th.zeros((B, 88), dtype=th.int64)
                init_onset_time = th.zeros((B, 88))
                init_onset_vel = th.zeros((B, 88))
            last_state = init_state.to(device)
            last_onset_time = init_onset_time.to(device)
            last_onset_vel = init_onset_vel.to(device)
            h = init_h
            h_vel = init_h_vel

            for t in range(T):
                c_fw = self.context_emb3(th.stack([last_state, last_onset_time, last_onset_vel], -1).unsqueeze(1)).squeeze(1)# B 88 4
                x_t = th.cat((x[:,:,t].permute(0,2,1), c_fw), dim=-1)  # B, 88, C
                x_t = x_t.reshape(B*88, -1)  # B*88, C
                x_t, h = self.lstm(x_t.unsqueeze(1), h) # x: B*88, 1, N
                out = self.output(x_t.reshape(B, 88, -1))  # B, 88, 5

                c_fw = self.context_emb3_v(th.stack([last_state, last_onset_time, last_onset_vel], -1).unsqueeze(1)).squeeze(1)# B 88 4
                x_t = th.cat((x_v[:,:,t].permute(0,2,1), c_fw), dim=-1)  # B, 88, C
                x_t = x_t.reshape(B*88, -1)  # B*88, C
                x_t, h_vel = self.lstm_v(x_t.unsqueeze(1), h_vel) # x: B*88, 1, N
                vel_out = self.output_vel(x_t.reshape(B, 88, -1))  # B, 88, 128

                arg_frame = out.argmax(-1)
                arg_vel = vel_out.argmax(-1)
                arg_vel = th.clamp(arg_vel, min=0, max=128)
                frame[:, t, target_pitch] = arg_frame[:, target_pitch]
                vel[:, t, target_pitch] = arg_vel[:, target_pitch]

                last_onset_time, last_onset_vel = update_context(last_onset_time, last_onset_vel, frame[:, t], vel[:, t])
                last_state = frame[:, t]

            return frame, vel

    def forward_mask(self, audio, c_mask, unpad_start=False, unpad_end=False):
        spec = self.frontend(audio)  # B, 3, F, T
        if unpad_start:
            spec = spec[:,:,:,self.n_fft//2//HOP:]
        if unpad_end:
            spec = spec[:,:,:,:-self.n_fft//2//HOP]
        B = spec.shape[0]
        T = spec.shape[3]
        x = self.front_block(spec.permute(0,1,3,2))  # B, C, T, 99
        c = self.context_emb(c_mask)  # B, T, 88, N
        x = self.cross_attention(x, c)
        c2 = self.context_emb2(c_mask)  # B, T, 88, N
        out = self.middle_block(x, c2)  # B C T 88
    
        x = self.front_block_v(spec.permute(0,1,3,2))  # B, C, T, 99
        c = self.context_emb_v(c_mask)  # B, T, 88, N
        x = self.cross_attention_v(x, c)
        c2 = self.context_emb2_v(c_mask)  # B, T, 88, N
        vel_out = self.middle_block_v(x, c2)  # B C T 88
        return out, vel_out 

        
    def update_context(self, last_onset_time, last_onset_vel, frame, vel):
        #  last_onset_time : 88
        #  last_onset_vel  : 88
        #  frame: 88
        #  vel  : 88
        
        onsets = (frame == 2) + (frame == 4)
        frames = (frame==2) + (frame == 3) + (frame == 4)

        cur_onset_time = th.zeros_like(last_onset_time)
        cur_onset_vel = th.zeros_like(last_onset_vel)

        onset_pos = onsets == 1
        frame_pos = (onsets != 1) * (frames == 1)

        cur_onset_time = onset_pos + frame_pos*(last_onset_time+1)
        cur_onset_vel = onset_pos*vel + frame_pos*last_onset_vel
        return cur_onset_time, cur_onset_vel


 

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


class ContextNetJoint(nn.Module):
    def __init__(self, n_hidden, out_dim=4):
        super().__init__()
        self.joint_net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )

        self.embedding = nn.Embedding(6, 2)
        self.concat_net = nn.Linear(n_hidden+2, out_dim)

    def forward(self, last, last_time, last_onset):
        joint_embed = self.joint_net(th.cat((last_time, last_onset), dim=-1))
        last = self.embedding(last)  # B x T x 88 x 5
        concat = th.cat((last, joint_embed), dim=-1)
        concat = self.concat_net(concat).permute(0, 1, 3, 2)
        return concat # B x T x out_dim x 88


class ContextNet(nn.Module):
    def __init__(self, bw=False, out_dim=16):
        super().__init__()
        n_hidden = 16
        input_dim = 4 if bw else 2
        self.emb_net = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )
        self.state_emb = nn.Embedding(6, 2)
        self.emb_proj = nn.Linear(n_hidden+2, out_dim)
    
    def forward(self, c):
        joint_emb = self.emb_net(c[:,:,:,1:].float())
        last = self.state_emb(c[:,:,:,0])
        c = th.cat((last, joint_emb), dim=-1) # B, T, 88, n_hidden+2
        c = self.emb_proj(c) # B, T, 88, 16
        return c


class CrossAttention(nn.Module):
    def __init__(self, cnn_unit) -> None:
        super().__init__()
    
        pos_enc = sinusoids(99, cnn_unit, max_timescale=100)
        pos_enc_c = sinusoids(88, cnn_unit, max_timescale=100)
        self.MHA = th.nn.MultiheadAttention(cnn_unit, num_heads=4, 
                                            kdim=cnn_unit, vdim=cnn_unit, dropout=0.1, batch_first=True)
        self.register_buffer('pos_enc', pos_enc)
        self.register_buffer('pos_enc_c', pos_enc_c)

    def forward(self, x, c):
        # x: B, C, T, 99
        # c: B, T, 88, C2
        B = x.shape[0]
        C = x.shape[1]
        T = x.shape[2]
        C2 = c.shape[3]
        c += self.pos_enc_c

        x = x.permute(0, 2, 3, 1) # B, T, 99, C
        x += self.pos_enc

        # Cross Attention
        x = x.reshape(B*T, 99, C)
        c = c.reshape(B*T, 88, C2)
        
        attn_output, _ = self.MHA(c, x, x)
        attn_output = attn_output.reshape(B, T, 88, C2)\
            .permute(0, 3, 1, 2) # B, C2, T, 88
        return attn_output
        




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

        pos_enc = sinusoids(88, cnn_unit, max_timescale=100)
        pos_enc_c = sinusoids(88, cnn_unit, max_timescale=100)
        self.register_buffer('pos_enc', pos_enc)
        self.register_buffer('pos_enc_c', pos_enc_c)
        self.hdc0 = HarmonicDilatedConv(n_input, cnn_unit, 1)
        self.hdc1 = HarmonicDilatedConv(cnn_unit, cnn_unit, 1)
        self.MHA = th.nn.MultiheadAttention(cnn_unit, num_heads=4,
                                            kdim=cnn_unit, vdim=cnn_unit, dropout=0.1, batch_first=True)
        self.hdc2 = HarmonicDilatedConv(cnn_unit, cnn_unit, 1)
        self.block = nn.Sequential(
            ConvFilmBlock(cnn_unit, cnn_unit, [3,1], 1, use_film=True, n_f=88),
            ConvFilmBlock(cnn_unit, cnn_unit, [3,1], 1, use_film=True, n_f=88),
        )
    
    def forward(self, x, c):
        # x: B x C x T x 88
        B = x.shape[0]
        C = x.shape[1]
        T = x.shape[2]
        C2 = c.shape[3]
        x = self.hdc0(x)
        x = self.hdc1(x)
        x = x.permute(0, 2, 3, 1).reshape(B*T, 88, C)
        c = c.reshape(B*T, 88, C2)

        x += self.pos_enc
        c += self.pos_enc_c
        attn_output, _ = self.MHA(c, x, x)
        attn_output = attn_output.reshape(B, T, 88, C2)\
            .permute(0, 3, 1, 2) # B, C, T, 99

        x = self.hdc2(attn_output)
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

        

    