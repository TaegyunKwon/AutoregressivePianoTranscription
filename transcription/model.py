import torch as th
from torch import nn
from torch.nn import functional as F
import nnAudio
from torchaudio import transforms

# from .cqt import CQT
from .constants import SR, HOP
from .context import random_modification, update_context

class ARModel(nn.Module):
    def __init__(self, config, perceptual_w=False):
        super().__init__()
        self.model = config.model
        self.win_fw = config.win_fw
        self.win_bw = config.win_bw
        self.n_fft = config.n_fft
        self.hidden_per_pitch = config.hidden_per_pitch
        self.context_len = self.win_fw + self.win_bw + 1


        # self.cqt = CQT(21, 8000, 4, 1, perceptual_w)
        self.melspectrogram = transforms.MelSpectrogram(sample_rate=SR, n_fft=config.n_fft,
            hop_length=HOP, f_min=config.f_min, f_max=config.f_max, n_mels=config.n_mels, normalized=False)

        if self.model == 'PAR':
            self.acoustic = SimpleConv2(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch)
            self.vel_acoustic = SimpleConv2(config.n_mels, config.cnn_unit, config.fc_unit, 
                                config.win_fw, config.win_bw, config.hidden_per_pitch)
        elif self.model == 'PC':
            self.acoustic = PC(config.n_mels, config.cnn_unit,
                                config.win_fw, config.win_bw, config.hidden_per_pitch)
            self.vel_acoustic = PC(config.n_mels, config.cnn_unit,
                                config.win_fw, config.win_bw, config.hidden_per_pitch)
        else:
            raise KeyError(f'wrong model:{self.model}')
            
        # self.context_net = ContextNet(config.hidden_per_pitch, out_dim=4)
        self.context_net = ContextNetJoint(config.hidden_per_pitch, out_dim=4)

        self.lstm = nn.LSTM(config.hidden_per_pitch+4, config.lstm_unit, num_layers=2, batch_first=False, bidirectional=False)
        self.output = nn.Linear(config.lstm_unit, 5)

        self.vel_lstm = nn.LSTM(config.hidden_per_pitch*2+4, config.lstm_unit, num_layers=2, batch_first=False, bidirectional=False)
        self.vel_output = nn.Linear(config.lstm_unit, 1)

    def forward(self, audio, last_states=None, last_onset_time=None, last_onset_vel=None, 
                init_state=None, init_onset_time=None, init_onset_vel=None, sampling='gt', 
                max_step=400, random_condition=False, return_softmax=False):
        if sampling == 'gt':
            batch_size = audio.shape[0]
            conv_out, vel_conv_out = self.local_forward(audio)  # B x T x hidden x 88
            n_frame = conv_out.shape[1] 

            if random_condition:
                last_states = random_modification(last_states, 0.1)

            context_enc = self.context_net(last_states, 
                                           last_onset_time.unsqueeze(-1), 
                                           last_onset_vel.unsqueeze(-1)) # B x T x out_dim x 88
            concat = th.cat((context_enc, conv_out), dim=2).\
                permute(1, 0, 3, 2).\
                    reshape(n_frame, batch_size*88, self.hidden_per_pitch+4)
            self.lstm.flatten_parameters()
            lstm_out, lstm_hidden = self.lstm(concat) # hidden_per_pitch
            frame_out = self.output(lstm_out) # n_frame, B*88 x n_class
            frame_out = frame_out.view(n_frame, batch_size, 88, 5).permute(1, 0, 2, 3) # B x n_frame x 88 x n_class

            vel_concat = th.cat((context_enc, conv_out.detach(), vel_conv_out), dim=2).\
                permute(1, 0, 3, 2).\
                    reshape(n_frame, batch_size*88, self.hidden_per_pitch*2+4)
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
                    '''
                    if step == 0:
                        conv_out, vel_conv_out = self.local_forward(
                            audio[:, : (offset + max_step) * HOP + self.n_fft//2],
                            unpad_end=True)
                    elif step == seg_edges[-1]:  # last segment
                        conv_out, vel_conv_out = self.local_forward(
                            audio[:, offset * HOP - self.n_fft//2 : ],
                            unpad_start=True)
                    else:
                        conv_out, vel_conv_out = self.local_forward(
                            audio[:, offset * HOP - self.n_fft//2: 
                                (offset + max_step) * HOP + self.n_fft//2],
                            unpad_start=True, unpad_end=True)
                    '''
                    if step == seg_edges[-1]:  # last segment
                        conv_out, vel_conv_out = self.local_forward(audio[:, offset * HOP: ])
                    else:
                        conv_out, vel_conv_out = self.local_forward(audio[:, offset * HOP : (offset + max_step + 10) * HOP])

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

        concat = th.cat((c, z), 2).\
                permute(1, 0, 3, 2).\
                    reshape(n_frame, batch_size*88, self.hidden_per_pitch+4)

        self.lstm.flatten_parameters()
        lstm_out, lstm_hidden = self.lstm(concat, h) # hidden_per_pitch
        frame_out = self.output(lstm_out) # n_frame, B*88 x n_class
        frame_out = frame_out.view(n_frame, batch_size, 88, 5).permute(1, 0, 2, 3) # B x n_frame x 88 x n_class

        vel_concat = th.cat((c, z, vel_z), dim=2).\
                        permute(1, 0, 3, 2).\
                            reshape(n_frame, batch_size*88, self.hidden_per_pitch*2+4)
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


'''
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
    def __init__(self, n_input, n_unit, n_f, hidden=16):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_unit, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(n_unit, n_unit, (3, 3), padding=1)
        self.bn = nn.BatchNorm2d(n_unit)
        self.film = FilmLayer(n_f, n_unit, hidden=hidden)

    def forward(self, x):
        # x : shape of B C F L
        x = F.relu(self.conv1(x))
        res = self.conv2(x)
        res = self.bn(res) 
        res = self.film(res.transpose(2,3)).transpose(2,3)
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
        '''
        fc_x = F.pad(fc_x, (self.win_bw, self.win_fw))
        fc_x = self.win_fc(fc_x).view(batch_size, self.hidden_per_pitch, 88, -1)
        fc_x = fc_x.permute(0, 3, 1, 2) # B T C 88
        '''
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


class SimpleConv2(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4),
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
    
    
class PC(nn.Module):
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_per_pitch = hidden_per_pitch
        self.win_bw = win_bw
        self.win_fw = win_fw

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, (7, 7), padding=3),
            nn.BatchNorm2d(cnn_unit),
            # FilmLayer(n_mels, hidden=16),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),

            nn.Dropout(0.25),

            nn.Conv2d(cnn_unit, cnn_unit, (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit),
            # FilmLayer(n_mels//4, hidden=16),
            nn.ReLU(),
            nn.Conv2d(cnn_unit, cnn_unit, (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit),
            # FilmLayer(n_mels//4, hidden=16),
            nn.ReLU(),
        )
        f_size = 40 
        self.large_conv_l1 = nn.Conv2d(cnn_unit, hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l3 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        self.film_1 = FilmLayer(n_mels//4, hidden=16)
        self.film_2 = FilmLayer(n_mels//4, hidden=16)
        self.film_3 = FilmLayer(n_mels//4, hidden=16)

        self.fc_1 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.window_cnn = nn.Sequential(
            nn.ZeroPad2d((0, 0, self.win_bw, self.win_fw)),
            nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (self.win_bw + self.win_fw + 1, 1))
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        batch_size = x.shape[0]
        x = self.cnn(x) # (B x H x L x n_mels/4)
        x = self.large_conv_l1(F.pad(x, (0,39)))
        x = F.relu(self.film_1(x))
        x = self.large_conv_l2(F.pad(x, (0,39)))
        x = F.relu(self.film_2(x))
        x = self.large_conv_l3(F.pad(x, (0,39)))
        x = F.relu(self.film_3(x))[:,:,:,:88]

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