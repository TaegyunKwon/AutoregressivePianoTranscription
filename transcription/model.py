import torch as th
from torch import nn
from torch.nn import functional as F
import nnAudio
from torchaudio import transforms

# from .cqt import CQT
from .constants import SR, HOP
from .context import random_modification, update_context
from .cqt import MultiCQT

class ARModel(nn.Module):
    def __init__(self, config, perceptual_w=False):
        super().__init__()
        self.model = config.model
        self.win_fw = config.win_fw
        self.win_bw = config.win_bw
        self.n_fft = config.n_fft
        self.hidden_per_pitch = config.hidden_per_pitch
        self.context_len = self.win_fw + self.win_bw + 1
        self.pitchwise = config.pitchwise_lstm
        if self.model == 'PAR_CQT':
            self.melspectrogram = MultiCQT()
        else:
            self.melspectrogram = transforms.MelSpectrogram(sample_rate=SR, n_fft=config.n_fft,
                hop_length=HOP, f_min=config.f_min, f_max=config.f_max, n_mels=config.n_mels, normalized=False)

        if self.model == 'PAR':
            self.acoustic = PAR(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
            self.vel_acoustic = PAR(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
        elif self.model == 'PAR_v2':
            self.acoustic = PAR_v2(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
            self.vel_acoustic = PAR_v2(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
        elif self.model == 'PAR_CQT':
            self.acoustic = PAR_CQT(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
            self.vel_acoustic = PAR_CQT(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
        elif self.model == 'PAR_CQT_v2':
            self.acoustic = PAR_CQT_v2(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch//2,
                                use_film=config.film)
            self.vel_acoustic = PAR_CQT_v2(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch//2,
                                use_film=config.film)
        elif self.model == 'PAR_CQT_v3':
            self.acoustic = PAR_CQT_v3(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch//2,
                                use_film=config.film)
            self.vel_acoustic = PAR_CQT_v3(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch//2,
                                use_film=config.film)
        elif self.model == 'PAR_org':
            self.acoustic = AllConv(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
            self.vel_acoustic = AllConv(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
        elif self.model == 'PC':
            self.acoustic = PC(config.n_mels, config.cnn_unit,
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
            self.vel_acoustic = PC(config.n_mels, config.cnn_unit,
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film)
        elif self.model == 'PC_v2':
            self.acoustic = PC(config.n_mels, config.cnn_unit,
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film, v2=True)
            self.vel_acoustic = PC(config.n_mels, config.cnn_unit,
                                config.win_fw, config.win_bw, config.hidden_per_pitch,
                                use_film=config.film, v2=True)
        elif self.model == 'PC_CQT':
            self.acoustic = PC_CQT(config.n_mels, config.cnn_unit,
                                config.win_fw, config.win_bw, config.hidden_per_pitch//2,
                                use_film=config.film)
            self.vel_acoustic = PC_CQT(config.n_mels, config.cnn_unit,
                                config.win_fw, config.win_bw, config.hidden_per_pitch//2,
                                use_film=config.film)
        else:
            raise KeyError(f'wrong model:{self.model}')
            
        # self.context_net = ContextNet(config.hidden_per_pitch, out_dim=4)
        self.context_net = ContextNetJoint(config.hidden_per_pitch, out_dim=4)
        if config.pitchwise_lstm:
            self.lstm = nn.LSTM(config.hidden_per_pitch+4, config.lstm_unit, num_layers=2, batch_first=False, bidirectional=False)
            self.output = nn.Linear(config.lstm_unit, 5)

            self.vel_lstm = nn.LSTM(config.hidden_per_pitch*2+4, config.lstm_unit, num_layers=2, batch_first=False, bidirectional=False)
            self.vel_output = nn.Linear(config.lstm_unit, 1)
        else:
            self.lstm = nn.LSTM((config.hidden_per_pitch+4)*88, config.lstm_unit, num_layers=2, batch_first=False, bidirectional=False)
            self.output = nn.Linear(config.lstm_unit, 88*5)

            self.vel_lstm = nn.LSTM((config.hidden_per_pitch*2+4)*88, config.lstm_unit, num_layers=2, batch_first=False, bidirectional=False)
            self.vel_output = nn.Linear(config.lstm_unit, 88)
            

    def forward(self, audio, last_states=None, last_onset_time=None, last_onset_vel=None, 
                init_state=None, init_onset_time=None, init_onset_vel=None, sampling='gt', 
                max_step=400, random_condition=False, return_softmax=False):
        if sampling == 'gt':
            batch_size = audio.shape[0]
            conv_out, vel_conv_out = self.local_forward(audio)  # B x T x hidden x 88
            n_frame = conv_out.shape[1] 

            # print(f'conv_out:{conv_out.shape}')
            if random_condition:
                last_states = random_modification(last_states, 0.1)

            context_enc = self.context_net(last_states, 
                                           last_onset_time.unsqueeze(-1), 
                                           last_onset_vel.unsqueeze(-1)) # B x T x out_dim x 88

            # print(f'context:{context_enc.shape}')
            if self.pitchwise:
                concat = th.cat((context_enc, conv_out), dim=2).\
                    permute(1, 0, 3, 2).reshape(n_frame, batch_size*88, self.hidden_per_pitch+4)
            else:
                concat = th.cat((context_enc, conv_out), dim=2).\
                    permute(1, 0, 3, 2).reshape(n_frame, batch_size, (self.hidden_per_pitch+4)*88)
            self.lstm.flatten_parameters()
            # print(f'concat:{concat.shape}')
            lstm_out, lstm_hidden = self.lstm(concat) # hidden_per_pitch
            # print(f'lstm:{lstm_out.shape}')
            frame_out = self.output(lstm_out) # n_frame, B*88 x n_class
            # print(f'frame_out:{frame_out.shape}')
            frame_out = frame_out.view(n_frame, batch_size, 88, 5).permute(1, 0, 2, 3) # B x n_frame x 88 x n_class

            if self.pitchwise:
                vel_concat = th.cat((context_enc, conv_out.detach(), vel_conv_out), dim=2).\
                    permute(1, 0, 3, 2).reshape(n_frame, batch_size*88, self.hidden_per_pitch*2+4)
            else:
                vel_concat = th.cat((context_enc, conv_out.detach(), vel_conv_out), dim=2).\
                    permute(1, 0, 3, 2).reshape(n_frame, batch_size, (self.hidden_per_pitch*2+4)*88)
            self.vel_lstm.flatten_parameters()
            vel_lstm_out, vel_lstm_hidden = self.vel_lstm(vel_concat) # hidden_per_pitch
            vel_out = self.vel_output(vel_lstm_out) # n_frame, B*88 x 1
            vel_out = vel_out.view(n_frame, batch_size, 88).permute(1, 0, 2)  # B x n_frame x 88

            if return_softmax:
                frame_out = F.log_softmax(frame_out, dim=-1)

            return frame_out, vel_out

        else:
            batch_size = audio.shape[0]
            audio_len = audio.shape[1]
            step_len = (audio_len - 1) // HOP+ 1
            device = audio.device
 
            n_segs = ((step_len - 1)//max_step + 1)
            if 0 <= audio_len - (n_segs-1)*max_step* HOP< self.n_fft//2: # padding size of cqt
                n_segs -= 1
            seg_edges = [el*max_step for el in range(n_segs)]

            if init_onset_time == None:
                init_state = th.zeros((batch_size, 88), dtype=th.int64)
                init_onset_time = th.zeros((batch_size, 88))
                init_onset_vel = th.zeros((batch_size, 88))
            last_state = init_state.to(device)
            last_onset_time = init_onset_time.to(device)
            last_onset_vel = init_onset_vel.to(device)

            context_enc = self.context_net(
                last_state.view(batch_size, 1, 88),
                last_onset_time.view(batch_size, 1, 88, 1),
                last_onset_vel.view(batch_size, 1, 88, 1))
            frame = th.zeros((batch_size, step_len, 88, 5)).to(device)
            vel = th.zeros((batch_size, step_len, 88)).to(device)

            c = context_enc[:,0:1]
            h, vel_h = None, None
            offset = 0

            for step in range(step_len):
                if step in seg_edges:
                    offset = step
                    if step == 0:  # First segment
                        unpad_start = False
                        start = 0
                    else:
                        del conv_out
                        del vel_conv_out
                        unpad_start = True
                        start = offset * HOP - self.n_fft//2 

                    if step == seg_edges[-1]:  # Last segment
                        unpad_end = False
                        end = None
                    else:
                        # margin for CNN
                        end = (offset + max_step + 10) * HOP + self.n_fft//2
                        unpad_end = True
                    
                    conv_out, vel_conv_out = self.local_forward(
                        audio[:, start: end],
                        unpad_start=unpad_start, unpad_end=unpad_end)

                frame_out, vel_out, h, vel_h = self.recurrent_step(
                    conv_out[:, step - offset].unsqueeze(1), 
                    vel_conv_out[:, step - offset].unsqueeze(1),
                    c, 
                    h, 
                    vel_h)
                frame[:, step] = frame_out.squeeze(1)
                vel[:, step] = vel_out.squeeze(1)

                arg_frame = th.argmax(frame_out[:,0], dim=-1)
                arg_vel = th.clamp(vel_out[:,0] * 128, min=0, max=128)
                cur_onset_time, cur_onset_vel = update_context(last_onset_time, last_onset_vel, arg_frame, arg_vel)
                last_onset_time = cur_onset_time
                last_onset_vel = cur_onset_vel 
                context_enc = self.context_net(
                    arg_frame.view(batch_size, 1, 88).to(audio.device),
                    last_onset_time.view(batch_size, 1, 88, 1).to(audio.device).div(156),
                    last_onset_vel.view(batch_size, 1, 88, 1).to(audio.device).div(128))
                c = context_enc

            if return_softmax:
                frame = F.log_softmax(frame, dim=-1)
            return frame, vel

    def local_forward(self, audio, unpad_start=False, unpad_end=False):
        mel = self.melspectrogram(
            audio[:, :-1]).transpose(-1, -2) # B L F
        mel = (th.log(th.clamp(mel, min=1e-9)) + 7) / 7
        if unpad_start:
            mel = mel[:,self.n_fft//2//HOP:]
        if unpad_end:
            mel = mel[:,:-self.n_fft//2//HOP]
        conv_out = self.acoustic(mel)  # B x T x hidden_per_pitch x 88
        vel_conv_out = self.vel_acoustic(mel) # B x T x hidden_per_pitch x 88

        return conv_out, vel_conv_out

    def recurrent_step(self, z, vel_z, c, h, vel_h):
        # z: B x T x hidden x 88
        batch_size = z.shape[0]
        n_frame = 1

        if self.pitchwise:
            concat = th.cat((c, z), 2).permute(1, 0, 3, 2).reshape(n_frame, batch_size*88, self.hidden_per_pitch+4)
        else:
            concat = th.cat((c, z), 2).permute(1, 0, 3, 2).reshape(n_frame, batch_size, (self.hidden_per_pitch+4)*88)

        self.lstm.flatten_parameters()
        lstm_out, lstm_hidden = self.lstm(concat, h) # hidden_per_pitch
        frame_out = self.output(lstm_out) # n_frame, B*88 x n_class
        frame_out = frame_out.view(n_frame, batch_size, 88, 5).permute(1, 0, 2, 3) # B x n_frame x 88 x n_class

        vel_concat = th.cat((c, z, vel_z), dim=2).permute(1, 0, 3, 2)
        if self.pitchwise:
            vel_concat = vel_concat.reshape(n_frame, batch_size*88, self.hidden_per_pitch*2+4)
        else:
            vel_concat = vel_concat.reshape(n_frame, batch_size, (self.hidden_per_pitch*2+4)*88)
        self.vel_lstm.flatten_parameters()
        vel_lstm_out, vel_lstm_hidden = self.vel_lstm(vel_concat, vel_h) # hidden_per_pitch
        vel_out = self.vel_output(vel_lstm_out) # n_frame, B*88 x 1
        vel_out = vel_out.view(n_frame, batch_size, 88).permute(1, 0, 2)  # B x n_frame x 88

        return frame_out, vel_out, lstm_hidden, vel_lstm_hidden

class ContextNetJoint(nn.Module):
    def __init__(self, n_hidden, out_dim=4):
        super().__init__()
        self.joint_net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )

        self.embedding = nn.Embedding(5, 2)

        self.concat_net = nn.Linear(n_hidden+2, out_dim)

    def forward(self, last, last_time, last_onset):
        joint_embed = self.joint_net(th.cat((last_time, last_onset), dim=-1))
        last = self.embedding(last)  # B x T x 88 x 5
        concat = th.cat((last, joint_embed), dim=-1)
        concat = self.concat_net(concat).permute(0, 1, 3, 2)
        return concat # B x T x out_dim x 88

class ContextNet(nn.Module):
    # Hard output states to continous values
    def __init__(self, n_hidden, out_dim=4):
        super().__init__()
        self.joint_net = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )

        self.embedding = nn.Embedding(5, 2)

        self.concat_net = nn.Sequential(
            nn.Linear(n_hidden+2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, last, last_time, last_vel):
        # last:      B x T x 88 x 5 (last states)
        # last_time: B x T x 88 x 1 (time passed after last onset, if it still exist)
        # last_vel:  B x T x 88 x 1 (velocity of last onset, if it still exist)
        joint_embed = self.joint_net(th.cat((last_time, last_vel), dim=-1))
        last = self.embedding(last)  # B x T x 88 x 2
        concat = th.cat((last, joint_embed), dim=-1)
        concat = self.concat_net(concat).transpose(2,3)
        return concat # B x T x out_dim x 88


class FilmLayer(nn.Module):
    def __init__(self, n_f, channel, hidden=16):
        super().__init__()
        pitch = (th.arange(n_f)/n_f).view(n_f, 1)
        self.register_buffer('pitch', pitch.float())
        self.alpha_linear = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channel, bias=False),
        )
        self.beta_linear = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channel, bias=False),
        )

    def forward(self, x):
        # x : shape of (B,C,L,F)
        alpha = self.alpha_linear(self.pitch) # F x C
        beta = self.beta_linear(self.pitch)
        
        x = x.permute(0,2,3,1) # (B, L, F, C)
        x = alpha * x + beta
        return x.permute(0, 3, 1, 2)


class FilmBlock(nn.Module):
    def __init__(self, n_input, n_unit, n_f, hidden=16, use_film=True):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_unit, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(n_unit, n_unit, (3, 3), padding=1)
        self.bn = nn.BatchNorm2d(n_unit)
        self.use_film = use_film
        if use_film:
            self.film = FilmLayer(n_f, n_unit, hidden=hidden)

    def forward(self, x):
        if self.use_film:
            # x : shape of B C F L
            x = F.relu(self.conv1(x))
            x = self.film(x.transpose(2,3)).transpose(2,3)
            res = self.conv2(x)
            res = self.bn(res) 
            res = self.film(res.transpose(2,3)).transpose(2,3)
            x = F.relu(x + res)
            return x
        else:
            x = F.relu(self.conv1(x))
            res = self.conv2(x)
            res = self.bn(res) 
            x = F.relu(x + res)
            return x
'''

class FilmBlock(nn.Module):
    def __init__(self, n_input, n_unit, n_f):
        super().__init__()

        pitch = (th.range(1, n_f)/n_f).view(1, n_f, 1)
        self.register_buffer('pitch', pitch.float())

        self.conv1 = nn.Conv2d(n_input, n_unit, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(n_unit, n_unit, (3, 3), padding=1)
        self.bn = nn.BatchNorm2d(n_unit)
        self.alpha_linear = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_unit),
        )
        self.beta_linear = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_unit),
        )

    def forward(self, x):
        # x : shape of B C F L
        batch_size = x.shape[0]
        alpha = self.alpha_linear(self.pitch)\
            .transpose(1,2).unsqueeze(-1)  # 1 C F 1
        beta = self.beta_linear(self.pitch)\
            .transpose(1,2).unsqueeze(-1)  # 1 C F 1

        x = F.relu(self.conv1(x))
        res = self.conv2(x)
        res = self.bn(res)
        res = alpha * res + beta
        x = F.relu(x + res)
        return x
class PAR(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * F
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        # self.win_fc = nn.Conv1d(fc_unit, hidden_per_pitch*88, (win_fw+win_bw+1))
        self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch*88)

        pitch = (th.range(1, 88)/88).view(1, 88, 1)
        self.register_buffer('pitch', pitch)
        self.pitch_cnn1 = nn.Conv2d(cnn_unit, hidden_per_pitch, (36, self.win_fw+self.win_bw+1))
        self.pitch_cnn2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (36, 1))
        self.film1 = FilmLayer(88)
        self.film2 = FilmLayer(88)
        self.layernorm = nn.LayerNorm([hidden_per_pitch*2, 88])

    def forward(self, cqt):
        batch_size = cqt.shape[0] # B, F, T
        x = cqt.unsqueeze(1)  # B 1 F T
        x = self.cnn(x)  # B C F T
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        fc_x = fc_x.transpose(1,2)  # B C T

        fc_x = F.pad(fc_x, (self.win_bw, self.win_fw)).unsqueeze(3)
        fc_x = F.unfold(fc_x, (self.win_bw + self.win_fw + 1, 1)) 
        fc_x = self.win_fc(fc_x.transpose(1,2))  # B L C
        fc_x = fc_x.view(batch_size, -1, self.hidden_per_pitch, 88)

        x = F.pad(x, (self.win_bw, self.win_fw, 0, 25))
        x = self.pitch_cnn1(x) # B C 88 T
        x = self.film1(x.transpose(2,3)).transpose(2,3)

        x = F.pad(x, (0, 0, 0, 35)) 
        x = self.pitch_cnn2(x) # B C 88 T
        x = self.film2(x.transpose(2,3)).transpose(2,3)

        x = x.permute(0, 3, 1, 2) # B T C 88

        x = th.cat((x, fc_x), -2)
        return F.relu(x)
'''


class PAR(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch*88)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        fc_x = fc_x.transpose(1,2)  # B C L
        fc_x = F.pad(fc_x, (self.win_bw, self.win_fw)).unsqueeze(3)
        fc_x = F.unfold(fc_x, (self.win_bw + self.win_fw + 1, 1)) 
        fc_x = self.win_fc(fc_x.transpose(1,2))  # B L C
        fc_x = fc_x.view(batch_size, -1, self.hidden_per_pitch, 88)

        x = self.layernorm(fc_x)
        return F.relu(x)
    
class PAR_v2(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.win_fc = nn.Conv1d(fc_unit, fc_unit, self.win_fw+self.win_bw+1)
        self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        fc_x = F.pad(fc_x.permute(0,2,1), (self.win_bw, self.win_fw)) # B C L 
        multistep_x = self.win_fc(fc_x)
        pitchwise_x = self.pitch_linear(multistep_x.transpose(1,2))
        pitchwise_x = pitchwise_x.view(batch_size, -1, self.hidden_per_pitch, 88)
        return F.relu(self.layernorm(pitchwise_x))
    
class PAR_CQT(nn.Module):
    # large conv - pitchwise fc model
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.large_conv = nn.Conv2d(cnn_unit, hidden_per_pitch, (49, self.win_fw+self.win_bw+1),
                                    stride=(2, 1))

        self.fc_1 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        x = F.pad(x, (self.win_bw, self.win_fw, 24, 3))
        x = self.large_conv(x) # B C 88, L
        x = x.view(x.shape[0], self.hidden_per_pitch*88, -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = x.view(x.shape[0], self.hidden_per_pitch, 88, -1).permute(0, 3, 1, 2)

        return F.relu(self.layernorm(x))

class PAR_CQT_v2(nn.Module):
    # two-path model
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.large_conv = nn.Conv2d(cnn_unit, hidden_per_pitch, (49, self.win_fw+self.win_bw+1),
                                    stride=(2, 1))
        self.pitch_fc_1 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.pitch_fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.pitch_fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.fc_1 = nn.Linear(790//4*cnn_unit, fc_unit)
        self.fc_2 = nn.Linear(fc_unit, hidden_per_pitch*88)

        self.layernorm = nn.LayerNorm([hidden_per_pitch*2, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        cnn_out = self.cnn(x)  # B C F L
        x = F.pad(cnn_out, (self.win_bw, self.win_fw, 24, 3))
        x = self.large_conv(x) # B H 88, L

        x_1 = x.view(batch_size, self.hidden_per_pitch*88, -1)
        x_1 = F.relu(self.pitch_fc_1(x_1))
        x_1 = F.relu(self.pitch_fc_2(x_1))
        x_1 = F.relu(self.pitch_fc_3(x_1))
        x_1 = x_1.view(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 3, 1, 2)

        x_2 = self.fc_1(cnn_out.permute(0,3,1,2).flatten(-2))
        x_2 = self.fc_2(x_2).view(batch_size, -1, self.hidden_per_pitch, 88)
        x = th.cat((x_1, x_2), -2)

        return F.relu(self.layernorm(x))

class PAR_CQT_v3(nn.Module):
    # two-path model
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.pitch_cnn1 = nn.Conv2d(cnn_unit, hidden_per_pitch, (49, self.win_fw+self.win_bw+1),
                                    stride=(2,1))
        self.pitch_cnn2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (49, 1))
        self.pitch_film1 = FilmLayer(88, hidden_per_pitch)
        self.pitch_film2 = FilmLayer(88, hidden_per_pitch)

        self.fc_1 = nn.Linear(790//4*cnn_unit, fc_unit)
        self.fc_2 = nn.Linear(fc_unit, hidden_per_pitch*88)

        self.layernorm = nn.LayerNorm([hidden_per_pitch*2, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        cnn_out = self.cnn(x)  # B C F L
        x = F.pad(cnn_out, (self.win_bw, self.win_fw, 24, 3))
        x = self.pitch_cnn1(x) # B H 88, L
        x = self.pitch_film1(x.transpose(2,3)).transpose(2,3)
        x = F.pad(x, (0, 0, 24, 24)) 
        x = self.pitch_cnn2(x)
        x = self.pitch_film2(x.transpose(2,3)) # B H L F 
        x = x.permute(0, 2, 1, 3)

        x_2 = self.fc_1(cnn_out.permute(0,3,1,2).flatten(-2))
        x_2 = self.fc_2(x_2).view(batch_size, -1, self.hidden_per_pitch, 88)
        x = th.cat((x, x_2), -2)

        return F.relu(self.layernorm(x))

class AllConv(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)

        self.pitch_cnn1 = nn.Conv2d(cnn_unit, hidden_per_pitch//2, (60, self.win_fw+self.win_bw+1))
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L

        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        fc_x = fc_x.transpose(1,2)  # B C L
        fc_x = F.pad(fc_x, (self.win_bw, self.win_fw)).unsqueeze(3)
        fc_x = F.unfold(fc_x, (self.win_bw + self.win_fw + 1, 1)) 
        fc_x = self.win_fc(fc_x.transpose(1,2))  # B L C
        fc_x = fc_x.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x = F.pad(x, (self.win_bw, self.win_fw, 59, 0))
        x = self.pitch_cnn1(x)[:,:,:88,:]
        x = x.permute(0, 3, 1, 2)

        x = th.cat((x, fc_x), -2)
        x = self.layernorm(x)
        return F.relu(x)

class PC(nn.Module):
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film=True, v2=False):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_per_pitch = hidden_per_pitch
        self.win_bw = win_bw
        self.win_fw = win_fw
        if v2:
            cnn_multipler = [4, 2, 1]
        else:
            cnn_multipler = [1, 1, 1]

        # input is batch_size * 1 channel * frames * input_features
        self.use_film = use_film
        if use_film:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                FilmLayer(n_mels, cnn_unit*cnn_multipler[0], hidden=16),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[1], hidden=16),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[2], hidden=16),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                nn.ReLU(),
            )
        f_size = 40 
        self.large_conv_l1 = nn.Conv2d(cnn_unit*cnn_multipler[2], hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l3 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        if use_film:
            self.film_1 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)
            self.film_2 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)
            self.film_3 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)

        self.fc_1 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.window_cnn = nn.Sequential(
            nn.ZeroPad2d((0, 0, self.win_bw, self.win_fw)),
            nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (self.win_bw + self.win_fw + 1, 1))
        )

    def forward(self, mel):
        # mel : B L F
        if self.use_film:
            x = mel.unsqueeze(1)
            batch_size = x.shape[0]
            x = self.cnn(x) # (B x H x L x n_mels/4)
            x = self.large_conv_l1(F.pad(x, (0,39)))
            x = F.relu(self.film_1(x))
            x = self.large_conv_l2(F.pad(x, (0,39)))
            x = F.relu(self.film_2(x))
            x = self.large_conv_l3(F.pad(x, (0,39)))
            x = F.relu(self.film_3(x))[:,:,:,:88]
        else:
            x = mel.unsqueeze(1)
            batch_size = x.shape[0]
            x = self.cnn(x) # (B x H x L x n_mels/4)
            x = self.large_conv_l1(F.pad(x, (0,39)))
            x = F.relu(x)
            x = self.large_conv_l2(F.pad(x, (0,39)))
            x = F.relu(x)
            x = self.large_conv_l3(F.pad(x, (0,39)))
            x = F.relu(x)[:,:,:,:88]
            
        x = x.transpose(2,3).reshape(batch_size, self.hidden_per_pitch*88, -1)
        x = self.fc_1(x)  # B x 1 x L x n_mels/4
        res = self.fc_2(F.relu(x))
        res = self.fc_3(F.relu(res))
        x = x + res
        x = x.reshape(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 1, 3, 2)
        # B, H, L, 88

        x = self.window_cnn(x)  # B x H x L x 88
        x = x.transpose(1, 2)
        # x = x.flatten(-2)

        return x  # B x L x H x 88

class PC_v3(nn.Module):
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film=True, v2=False):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_per_pitch = hidden_per_pitch
        self.win_bw = win_bw
        self.win_fw = win_fw
        if v2:
            cnn_multipler = [4, 2, 1]
        else:
            cnn_multipler = [1, 1, 1]

        # input is batch_size * 1 channel * frames * input_features
        self.use_film = use_film
        if use_film:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                FilmLayer(n_mels, cnn_unit*cnn_multipler[0], hidden=16),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[1], hidden=16),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[2], hidden=16),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                nn.ReLU(),
            )
        f_size = 40 
        self.large_conv_l1 = nn.Conv2d(cnn_unit*cnn_multipler[2], hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l3 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        if use_film:
            self.film_1 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)
            self.film_2 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)
            self.film_3 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)

        self.fc_1 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.window_cnn = nn.Sequential(
            nn.ZeroPad2d((0, 0, self.win_bw, self.win_fw)),
            nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (self.win_bw + self.win_fw + 1, 1))
        )

    def forward(self, mel):
        if self.use_film:
            x = mel.unsqueeze(1)
            batch_size = x.shape[0]
            x = self.cnn(x) # (B x H x L x n_mels/4)
            x = self.large_conv_l1(F.pad(x, (0,39)))
            x = F.relu(self.film_1(x))
            x = self.large_conv_l2(F.pad(x, (0,39)))
            x = F.relu(self.film_2(x))
            x = self.large_conv_l3(F.pad(x, (0,39)))
            x = F.relu(self.film_3(x))[:,:,:,:88]
        else:
            x = mel.unsqueeze(1)
            batch_size = x.shape[0]
            x = self.cnn(x) # (B x H x L x n_mels/4)
            x = self.large_conv_l1(F.pad(x, (0,39)))
            x = F.relu(x)
            x = self.large_conv_l2(F.pad(x, (0,39)))
            x = F.relu(x)
            x = self.large_conv_l3(F.pad(x, (0,39)))
            x = F.relu(x)[:,:,:,:88]
            
        x = x.transpose(2,3).reshape(batch_size, self.hidden_per_pitch*88, -1)
        x = self.fc_1(x)  # B x 1 x L x n_mels/4
        res = self.fc_2(F.relu(x))
        res = self.fc_3(F.relu(res))
        x = x + res
        x = x.reshape(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 1, 3, 2)
        # B, H, L, 88

        x = self.window_cnn(x)  # B x H x L x 88
        x = x.transpose(1, 2)
        # x = x.flatten(-2)

        return x  # B x L x H x 88



class PC_CQT(nn.Module):
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film=True):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_per_pitch = hidden_per_pitch
        self.win_bw = win_bw
        self.win_fw = win_fw
        cnn_multipler = [4, 2, 1]

        # input is batch_size * 1 channel * frames * input_features
        self.use_film = use_film
        if use_film:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                FilmLayer(n_mels, cnn_unit*cnn_multipler[0], hidden=16),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[1], hidden=16),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[2], hidden=16),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                nn.ReLU(),
            )
        f_size = 49 
        self.large_conv_l1 = nn.Conv2d(cnn_unit*cnn_multipler[2], hidden_per_pitch, (1, f_size), padding=0, stride=(1,2))
        self.large_conv_l2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l3 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        if use_film:
            self.film_1 = FilmLayer(88, hidden_per_pitch, hidden=16)
            self.film_2 = FilmLayer(88, hidden_per_pitch, hidden=16)
            self.film_3 = FilmLayer(88, hidden_per_pitch, hidden=16)
       
        self.fc_0 = nn.Sequential(
            nn.Conv2d(cnn_unit*cnn_multipler[2], 4, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1, padding=0),
            nn.ReLU(),
        )
        self.fc_1 = nn.Conv1d(790//4, hidden_per_pitch*88, 1, padding=0)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.window_cnn = nn.Sequential(
            nn.ZeroPad2d((0, 0, self.win_bw, self.win_fw)),
            nn.Conv2d(hidden_per_pitch*2, hidden_per_pitch*2, (self.win_bw + self.win_fw + 1, 1))
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        batch_size = x.shape[0]
        cnn_out = self.cnn(x) # (B x H x L x n_mels/4)
        if self.use_film:
            x = self.large_conv_l1(F.pad(cnn_out, (24,3)))
            x = F.relu(self.film_1(x))
            x = self.large_conv_l2(F.pad(x, (24,24)))
            x = F.relu(self.film_2(x))
            x = self.large_conv_l3(F.pad(x, (24,24)))
            x = F.relu(self.film_3(x))
        else:
            x = self.large_conv_l1(F.pad(cnn_out, (24,3)))
            x = F.relu(x)
            x = self.large_conv_l2(F.pad(x, (24,24)))
            x = F.relu(x)
            x = self.large_conv_l3(F.pad(x, (24,24)))
            x = F.relu(x)
        x_pitchwise = self.fc_0(cnn_out)  # B x 1 x L x n_mels/4
        x_pitchwise = x_pitchwise.transpose(1, 2).flatten(-2).transpose(1,2) # (B x n_mels/4 x L)

        x_pitchwise = F.relu(self.fc_1(x_pitchwise)) # B x H*88 x L
        res = self.fc_2(x_pitchwise)
        res = self.fc_3(F.relu(res))
        x_pitchwise = x_pitchwise + res
        x_pitchwise = x_pitchwise.reshape(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 1, 3, 2)
        # B, H, L, 88

        x = th.cat((x_pitchwise, x), 1)
        x = self.window_cnn(x)  # B x H x L x 88
        x = x.transpose(1, 2)
        # x = x.flatten(-2)

        return x  # B x L x H x 88