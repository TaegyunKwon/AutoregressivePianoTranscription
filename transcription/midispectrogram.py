import math
import warnings
from typing import Callable, Optional

import torch
from torch import Tensor
from torchaudio import functional as F
from torchaudio import transforms
import librosa


class MIDISpectrogram(torch.nn.Module):
    __constants__ = ["sample_rate", "n_fft", "win_length", "hop_length", "pad", "n_mels_per_semitone"]
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        midi_min: int = 21,
        midi_max: int = 108,
        pad: int = 0,
        n_mels_per_semitone: int = 4,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        norm: Optional[str] = None,
    ) -> None:
        super(MIDISpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels_per_semitone = n_mels_per_semitone  # number of midi bins per semitone
        self.midi_max = midi_max
        self.midi_min = midi_min
        self.spectrogram = transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )
        self.midi_scale = MIDIScale(
            self.n_mels_per_semitone, self.sample_rate, self.midi_min, self.midi_max, self.n_fft // 2 + 1, norm
        )
        
    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveform)
        midi_specgram = self.midi_scale(specgram)
        return midi_specgram


class CombinedSpec(torch.nn.Module):
    # class for combining spectrogram and midi spectrogram
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 4096,
        hop_length: int = 512,
        midi_min_low: int = 41, # librosa.midi_to_hz(41) = 87.31 Hz (F2)
        midi_max_low: int = 59, # librosa.midi_to_hz(119) = 7902.1 (B8)
        midi_min_high: int = 60, # librosa.midi_to_hz(41) = 87.31 Hz (F2)
        midi_max_high: int = 118, # librosa.midi_to_hz(118) = 7458.6, 119: 7902.1 (B8)
        spec_low: int = 11, # number of bins we will used in spectrogram. 12*(16000/2048)=85.94 Hz
        n_per_semi_low: int = 3,
        n_per_semi_high: int = 8,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        norm: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = 0
        self.power = power
        self.normalized = normalized
        self.midi_min_low = midi_min_low
        self.midi_max_low = midi_max_low
        self.midi_min_high = midi_min_high
        self.midi_max_high = midi_max_high
        self.spec_low = spec_low
        self.n_per_semi_low = n_per_semi_low
        self.n_per_semi_high = n_per_semi_high
        self.spectrogram = transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )
        self.midi_scale_low = MIDIScale(
            self.n_per_semi_low, self.sample_rate, self.midi_min_low, self.midi_max_low, self.n_fft // 2 + 1, norm
        )
        self.midi_scale_high = MIDIScale(
            self.n_per_semi_high, self.sample_rate, self.midi_min_high, self.midi_max_high, self.n_fft // 2 + 1, norm
        )
        
    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveform)
        midi_spec_low = self.midi_scale_low(specgram)
        midi_spec_high = self.midi_scale_high(specgram)
        return midi_spec_low, midi_spec_high, specgram[:, 1:1+self.spec_low]
        

class MIDIScale(torch.nn.Module):
    __constants__ = ["sample_rate", "midi_min", "midi_max"]
    def __init__(
        self,
        n_mels_per_semitone: int = 4,
        sample_rate: int = 16000,
        midi_min: int = 21,
        midi_max: int = 108,
        n_stft: int = 201,
        norm: Optional[str] = None,
    ) -> None:
        super(MIDIScale, self).__init__()
        self.n_mels_per_semitone = n_mels_per_semitone
        self.sample_rate = sample_rate
        self.midi_max = midi_max if midi_max is not None else 108
        self.midi_min = midi_min
        self.norm = norm

        fb = midiscale_fbanks(n_stft, self.midi_min, self.midi_max, self.n_mels_per_semitone, self.sample_rate, self.norm)
        self.register_buffer("fb", fb)
    def forward(self, specgram: Tensor) -> Tensor:

        # (..., time, freq) dot (freq, n_midis) -> (..., n_midis, time)
        midi_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

        return midi_specgram
    
def midiscale_fbanks(
    n_freqs: int,
    midi_min: int,
    midi_max: int,
    n_mels_per_semitone: int,
    sample_rate: int,
    norm: Optional[str] = None,
) -> Tensor:
    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")
    n_mels = n_mels_per_semitone * (midi_max - midi_min + 1)
    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate midi freq bins
    m_width = 1/n_mels_per_semitone
    m_min = midi_min - 0.5 - m_width/2
    m_max = midi_max + 0.5 + m_width/2

    m_pts = torch.linspace(m_min, m_max, n_mels+2)
    f_pts = _midi_to_hz(m_pts)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb

def _hz_to_midi(freq):
    return 12 * (torch.log2(freq) - torch.log2(440.0)) + 69

def _midi_to_hz(notes):
    return 440.0 * (2.0 ** ((notes - 69.0) / 12.0))

def _create_triangular_filterbank(
    all_freqs: Tensor,
    f_pts: Tensor,
) -> Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb
