# to make midi-centered cqt
import torch as th
import torch.nn as nn
import librosa
import nnAudio.features
import nnAudio.utils
from .constants import SR, HOP
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

midi_min = 21
n_per_midi = 4
Q = 1

midi_width = 1 / n_per_midi
if n_per_midi % 2 == 0:
    midi_start = midi_min - (n_per_midi//2 - 0.5)*midi_width
else:
    midi_start = midi_min - (n_per_midi//2)*midi_width

freq_start = 440*2**((midi_start-69)/12)

cqt_kernel = nnAudio.utils.create_cqt_kernels(
    Q=Q,
    fs=SR,
    fmin=freq_start,
    bins_per_octave=48,
    fmax=8000
    )
kernel_len = cqt_kernel[1]
freqs = cqt_kernel[3]

# https://librosa.org/doc/0.8.1/generated/librosa.perceptual_weighting.html#librosa.perceptual_weighting
perceptual_offset = librosa.convert.frequency_weighting(freqs, kind='A').reshape((-1, 1))


class CQT(nn.Module):
    def __init__(self, midi_min, f_max, n_per_midi, Q, perceptual_weighting=False):
        super().__init__()
        midi_width = 1 / n_per_midi
        if n_per_midi % 2 == 0:
            midi_start = midi_min - (n_per_midi//2 - 0.5)*midi_width
        else:
            midi_start = midi_min - (n_per_midi//2)*midi_width

        freq_start = 440*2**((midi_start-69)/12)
        self.cqt = nnAudio.features.cqt.CQT(
            sr=SR, hop_length=HOP, fmin=freq_start, fmax=f_max, 
            bins_per_octave=n_per_midi*12, filter_scale=Q)

        cqt_kernel = nnAudio.utils.create_cqt_kernels(
            Q=Q,
            fs=SR,
            fmin=freq_start,
            bins_per_octave=48,
            fmax=8000
            )
        self.kernel_len = cqt_kernel[1]
        self.freqs = cqt_kernel[3]
        assert kernel_len//2 % HOP == 0, f'unpad is not possible with kernel_len:{self.kernel_len}'
        
        self.perceptual_weighting=perceptual_weighting
        if perceptual_weighting:
            # https://librosa.org/doc/0.8.1/generated/librosa.perceptual_weighting.html#librosa.perceptual_weighting
            self.perceptual_offset = librosa.convert.frequency_weighting(freqs, kind='A').reshape((-1, 1))/20

    def forward(self, audio, unpad_start, unpad_end):
        # audio: (B x L)
        # cqt: (B, F, T)
        cqt = self.cqt(
            audio.reshape(-1, audio.shape[-1])[:, :-1])
        if unpad_start:
            cqt = cqt[:,:,self.kernel_len//2//HOP:]
        if unpad_end:
            cqt = cqt[:,:,:-self.kernel_len//2//HOP]
        if self.perceptual_weighting:
            cqt = th.log(th.clamp(cqt, min=1e-6)+7+perceptual_offset)
        else:
            cqt = th.log(th.clamp(cqt, min=1e-6)+7)
        return cqt
            
class MultiCQT(nn.Module):
    def __init__(self):
        super().__init__()
        midi_low = 21
        midi_mid = 57
        midi_high = 75
        midi_edges = [midi_low, midi_mid, midi_high]
        n_per_midi = 8

        midi_width = 1 / n_per_midi
        edge_freq = []
        for edge in midi_edges:
            midi_start = edge - (n_per_midi//2 - 0.5)*midi_width
            freq_start = 440*2**((midi_start-69)/12)
            edge_freq.append(freq_start)

        self.cqt_low = nnAudio.features.cqt.CQT(sr=16000, hop_length=512, fmin=edge_freq[0], fmax=edge_freq[1]-0.1, bins_per_octave=96,
                                        filter_scale=0.5)
        self.cqt_mid = nnAudio.features.cqt.CQT(sr=16000, hop_length=512, fmin=edge_freq[1], fmax=edge_freq[2]-0.1, bins_per_octave=96,
                                        filter_scale=1)
        self.cqt_high = nnAudio.features.cqt.CQT(sr=16000, hop_length=512, fmin=edge_freq[2], fmax=8000, bins_per_octave=96,
                                        filter_scale=1.5)

    def forward(self, audio):
        # audio: (B x L)
        # cqt: (B, F, T)
        cqts = [self.cqt_low(audio), self.cqt_mid(audio), self.cqt_high(audio)]
        return th.concat(cqts, 1)