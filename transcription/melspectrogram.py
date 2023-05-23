import torch 
from torch import nn

class MelSpectrogram(nn.Module):
    """This function is to calculate the Melspectrogram of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``
    The correct shape will be inferred automatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.
    Parameters
    ----------
    sr : int
        The sampling rate for the input audio.
        It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.
    n_fft : int
        The window size for the STFT. Default value is 2048
    win_length : int
        the size of window frame and STFT filter.
        Default: None (treated as equal to n_fft)
    n_mels : int
        The number of Mel filter banks. The filter banks maps the n_fft to mel bins.
        Default value is 128.
    hop_length : int
        The hop (or stride) size. Default value is 512.
    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.
    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``,
        the time index is the beginning of the STFT kernel, if ``True``, the time index is the
        center of the STFT kernel. Default value if ``True``.
    pad_mode : str
        The padding method. Default value is 'reflect'.
    htk : bool
        When ``False`` is used, the Mel scale is quasi-logarithmic. When ``True`` is used, the
        Mel scale is logarithmic. The default value is ``False``.
    fmin : int
        The starting frequency for the lowest Mel filter bank.
    fmax : int
        The ending frequency for the highest Mel filter bank.
    norm :
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization, AKA 'slaney' default in librosa).
        Otherwise, leave all the triangles aiming for
        a peak value of 1.0
    trainable_mel : bool
        Determine if the Mel filter banks are trainable or not. If ``True``, the gradients for Mel
        filter banks will also be calculated and the Mel filter banks will be updated during model
        training. Default value is ``False``.
    trainable_STFT : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``.
    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.
    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)``.
    Examples
    --------
    >>> spec_layer = Spectrogram.MelSpectrogram()
    >>> specs = spec_layer(x)
    """

    def __init__(
        self,
        sr=22050,
        n_fft=2048,
        win_length=None,
        n_mels=128,
        hop_length=512,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        htk=False,
        fmin=0.0,
        fmax=None,
        norm=1,
        logscale=False,
        trainable_mel=False,
        trainable_STFT=False,
        verbose=True,
        **kwargs
    ):

        super().__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.power = power
        self.trainable_mel = trainable_mel
        self.trainable_STFT = trainable_STFT

        # Preparing for the stft layer. No need for center
        self.stft = STFT(
            n_fft=n_fft,
            win_length=win_length,
            freq_bins=None,
            hop_length=hop_length,
            window=window,
            freq_scale="no",
            center=center,
            pad_mode=pad_mode,
            sr=sr,
            trainable=trainable_STFT,
            output_format="Magnitude",
            verbose=verbose,
            **kwargs
        )

        # Create filter windows for stft
        start = time()

        # Creating kernel for mel spectrogram
        start = time()
        mel_basis = get_mel(sr, n_fft, n_mels, fmin, fmax, htk=htk, norm=norm, logscale=logscale)
        mel_basis = torch.tensor(mel_basis)

        if verbose == True:
            print(
                "STFT filter created, time used = {:.4f} seconds".format(time() - start)
            )
            print(
                "Mel filter created, time used = {:.4f} seconds".format(time() - start)
            )
        else:
            pass

        if trainable_mel:
            # Making everything nn.Parameter, so that this model can support nn.DataParallel
            mel_basis = nn.Parameter(mel_basis, requires_grad=trainable_mel)
            self.register_parameter("mel_basis", mel_basis)
        else:
            self.register_buffer("mel_basis", mel_basis)

        # if trainable_mel==True:
        #     self.mel_basis = nn.Parameter(self.mel_basis)
        # if trainable_STFT==True:
        #     self.wsin = nn.Parameter(self.wsin)
        #     self.wcos = nn.Parameter(self.wcos)

    def forward(self, x):
        """
        Convert a batch of waveforms to Mel spectrograms.
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        x = broadcast_dim(x)

        spec = self.stft(x, output_format="Magnitude") ** self.power

        melspec = torch.matmul(self.mel_basis, spec)
        return melspec

    def extra_repr(self) -> str:
        return "Mel filter banks size = {}, trainable_mel={}".format(
            (*self.mel_basis.shape,), self.trainable_mel, self.trainable_STFT
        )


def get_mel(
    sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1, dtype=np.float32
):
    """
    This function is cloned from librosa 0.7.
    Please refer to the original
    `documentation <https://librosa.org/doc/latest/generated/librosa.filters.mel.html>`__
    for more info.
    Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft     : int > 0 [scalar]
        number of FFT components
    n_mels    : int > 0 [scalar]
        number of Mel bands to generate
    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`
    htk       : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.
    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix
    Notes
    -----
    This function caches at level 10.
    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])
    Clip the maximum frequency to 8KHz
    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    >>> plt.show()
    """

    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError("Unsupported norm: {}".format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels."
        )

    return weights