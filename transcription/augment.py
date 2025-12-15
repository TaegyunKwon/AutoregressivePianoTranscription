import os
import random
import copy
from pathlib import Path
import numpy as np
from audiomentations import AddGaussianSNR, Compose, PitchShift, AddShortNoises, ApplyImpulseResponse, AddBackgroundNoise, SevenBandParametricEQ, PolarityInversion, Reverse

class AugmentatorAudiomentations:
    def __init__(self,
                 sampleRate = 16000,
                 pitchShiftRange = (-0.2, 0.2),
                 eqDBRange =  (-3, 3),
                 snrRange = (3, 40),
                 convIRFolder = None,
                 noiseFolder = None,
                 ):

        transformList = [
                    PitchShift(*pitchShiftRange, p = 0.5),
                    SevenBandParametricEQ( *eqDBRange, p = 0.5)
                ]

        self.transform = Compose(transformList)
        
        
        if convIRFolder is not None:
            irPath = Path(convIRFolder)
            fileList = list(irPath.glob(os.path.join('**','*.flac')))
            self.reverb = ApplyImpulseResponse(fileList, p = 0.5, lru_cache_size = 2000, leave_length_unchanged = True)
            print("aug: convIR enabled")
        else:
            self.reverb = None

        transformNoiseList = []
        if noiseFolder is not None:
            noisePath = Path(noiseFolder)
            fileList = list(noisePath.glob(os.path.join('**','*.flac')))

            noiseTrans = Compose([ 
                    PolarityInversion(),
                    Reverse(),
                    ])

            transformNoiseList.append( 
                    AddBackgroundNoise(
                        fileList,
                        min_snr_db=snrRange[0],
                        max_snr_db=snrRange[1],
                        p = 0.5,
                        noise_transform = noiseTrans))

            print("aug: noise enabled")

        transformNoiseList.append(
            AddGaussianSNR(
                    min_snr_db = snrRange[0],
                    max_snr_db = snrRange[1],
                    p = 0.5)
            )

        self.transformNoise = Compose(transformNoiseList)
        self.sampleRate = sampleRate

    def __call__(self, x):


        x = copy.deepcopy(x)

        # randomly downmix channels
        if len(x.shape) == 2:
            nChannel = x.shape[0]

            weight = 2*np.random.rand(1, nChannel)-1
            weight = (weight+1e-8)/(np.sum(np.abs(weight))+1e-8)

            x = np.matmul(weight, x)
            x = x.astype(np.float32)

        x = self.transform(x, sample_rate = self.sampleRate)

        # apply transform before impulse response
        if self.reverb is not None:
            xReverb = self.reverb(x, sample_rate = self.sampleRate)
            
            # randomize the wet/dry ratio
            alpha = random.random()

            x = alpha*x + (1-alpha)*xReverb



        x = self.transformNoise(x, sample_rate = self.sampleRate)


        return x
