from torch import nn
import torch as th
from .constants import HOP
from collections import defaultdict
from torch.nn import functional as F
from .model import HarmonicDilatedConv, MIDIFrontEnd
from .data import MAESTRO_V3
from torch.utils.data import DataLoader
from tqdm import tqdm

class ARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = 495
        self.hidden_per_pitch = 48
        self.frontend = MIDIFrontEnd(5)
        self.acoustic = PAR_v2_HPP(48, 48, 5)

    def forward(self, audio, max_step = 400):
        batch_size = audio.shape[0]
        audio_len = audio.shape[1]
        step_len = (audio_len - 1) // HOP+ 1

        n_segs = ((step_len - 1)//max_step + 1)
        if 0 <= audio_len - (n_segs-1)*max_step* HOP< self.n_fft//2: # padding size of cqt
            n_segs -= 1
        seg_edges = [el*max_step for el in range(n_segs)]

        frame = th.zeros((batch_size, step_len, 128, 88))

        offset = 0

        # print(f'n_segs:{n_segs}, n_step:{step_len}')
        seg_num = 0
        for step in seg_edges:
            # print(f'segment: {seg_num}')
            seg_num += 1
            offset = step
            if step == 0:  # First segment
                unpad_start = False
                start = 0
            else:
                del conv_out
                unpad_start = True
                start = offset * HOP - self.n_fft//2 

            if step == seg_edges[-1]:  # Last segment
                unpad_end = False
                end = None
            else:
                # margin for CNN
                end = (offset + max_step + 10) * HOP + self.n_fft//2
                unpad_end = True
            
            conv_out = self.local_forward(
                audio[:, start: end],
                unpad_start=unpad_start, unpad_end=unpad_end)
            if step == seg_edges[-1]:
                frame[:, offset:] = conv_out.detach().cpu()
            else:
                frame[:, offset:offset+max_step] = conv_out[:, :max_step].detach().cpu()
        return frame

    def local_forward(self, audio, unpad_start=False, unpad_end=False):
        spec = self.frontend(audio)
        if unpad_start:
            spec = spec[:,:,:,self.n_fft//2//HOP:]
        if unpad_end:
            spec = spec[:,:,:,:-self.n_fft//2//HOP]
        conv_out = self.acoustic(spec)  # B x T x hidden_per_pitch x 88
        return conv_out

class PAR_v2_HPP(nn.Module):
    def get_conv2d_block(self, channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1]):
        if(pool_size == None):
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out),
                
            )
        else:
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out)
            )

    # SimpleConv without Pitchwise Conv
    def __init__(self, cnn_unit, hidden_per_pitch, n_per_pitch=5):
        super().__init__()
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.block_1 = self.get_conv2d_block(3, cnn_unit, kernel_size=7)
        self.block_2 = self.get_conv2d_block(cnn_unit, cnn_unit, kernel_size=7)
        self.block_2_5 = self.get_conv2d_block(cnn_unit, cnn_unit, kernel_size=7)

        c3_out = 128
        
        self.conv_3 = HarmonicDilatedConv(cnn_unit, c3_out, n_per_pitch)
        self.conv_4 = HarmonicDilatedConv(c3_out, c3_out, n_per_pitch)
        self.conv_5 = HarmonicDilatedConv(c3_out, c3_out, n_per_pitch)

        self.block_4 = self.get_conv2d_block(c3_out, c3_out, pool_size=[1, n_per_pitch], dilation=[1, 12*n_per_pitch])
        self.block_5 = self.get_conv2d_block(c3_out, c3_out, dilation=[1, 12])
        self.block_6 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_7 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_8 = self.get_conv2d_block(c3_out, c3_out, [5,1])

    def forward(self, mel):
        x = self.block_1(mel.permute(0, 1, 3, 2))
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.block_4(x)
        x = x[:,:,:,:88]
        # => [b x 1 x T x 88]

        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x
        # x = self.conv_9(x)
        # x = torch.relu(x)
        # x = self.conv_10(x)
        # x = torch.sigmoid(x)
        
        x = x.permute(0, 2, 1, 3)  # B, 128, T, 88 -> B, T, 128, 88
        return x  


class PadCollate:
    def __call__(self, data):
        max_len = data[0]['audio'].shape[0] // HOP
        
        for datum in data:
            step_len = datum['audio'].shape[0] // HOP
            datum['step_len'] = step_len
            pad_len = max_len - step_len
            pad_len_sample = pad_len * HOP
            datum['audio'] = F.pad(datum['audio'], (0, pad_len_sample))

        batch = defaultdict(list)
        for key in data[0].keys():
            if key == 'audio':
                batch[key] = th.stack([datum[key] for datum in data], 0)
            else :
                batch[key] = [datum[key] for datum in data]
        return batch
    

def load_pretrain():
    model = ARModel()
    ckp = th.load('model_130k_0.9140.pt')
    model.load_state_dict(ckp['model_state_dict'], strict=False)
    return model