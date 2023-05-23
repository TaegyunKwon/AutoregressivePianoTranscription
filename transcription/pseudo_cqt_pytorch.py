cqt_filter_fft = librosa.constantq.__cqt_filter_fft

class PseudoCqt():
    """A class to compute pseudo-CQT with Pytorch.
    Written by Keunwoo Choi
    API (+implementations) follows librosa (https://librosa.github.io/librosa/generated/librosa.core.pseudo_cqt.html)
    
    Usage:
        src, _ = librosa.load(filename)
        src_tensor = torch.tensor(src)
        cqt_calculator = PseudoCqt()
        cqt_calculator(src_tensor)
        
    """
    def __init__(self, sr=22050, hop_length=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=0.0, filter_scale=1,
               norm=1, sparsity=0.01, window='hann', scale=True,
               pad_mode='reflect'):
        
        assert scale
        assert window == "hann"
        if fmin is None:
            fmin = 2 * 32.703195 # note_to_hz('C2') because C1 is too low

        if tuning is None:
            tuning = 0.0  # let's make it simple
        
        fft_basis, n_fft, _ = cqt_filter_fft(sr, fmin, n_bins, bins_per_octave,
                                               tuning, filter_scale, norm, sparsity,
                                               hop_length=hop_length, window=window)

        fft_basis = np.abs(fft_basis.astype(dtype=npdtype)).todense()  # because it was sparse. (n_bins, n_fft)
        self.fft_basis = torch.tensor(fft_basis)  # (n_freq, n_bins)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.scale = scale
        self.window = torch.hann_window(self.n_fft)
        self.npdtype = np.float32
    
    def __call__(self, y):
        return self.forward(y)
    
    def forward(self, y):
        D_torch = torch.stft(y, self.n_fft, 
                             hop_length=self.hop_length,
                            window=self.window).pow(2).sum(-1)  # n_freq, time
        D_torch = torch.sqrt(D_torch + EPS)  # without EPS, backpropagating through CQT can yield NaN.
        # Project onto the pseudo-cqt basis
        C_torch = torch.matmul(self.fft_basis, D_torch)  # n_bins, time
     
        C_torch /= torch.tensor(np.sqrt(self.n_fft))  # because `scale` is always True
        return to_decibel(C_torch)