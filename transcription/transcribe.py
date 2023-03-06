from collections import defaultdict
import math
import tempfile
from pathlib import Path
import soundfile
import subprocess
from transcription.constants import HOP
import torch as th
import torch.nn.functional as F


def load_audio(audiofile):
    try:
        audio, sr = soundfile.read(audiofile)
        if audio.shape[1] != 1:
            raise Exception
        if sr != 16000:
            raise Exception
    except:
        path_audio = Path(audiofile)
        filetype = path_audio.suffix
        assert filetype in ['.mp3', '.ogg', '.flac', '.wav', '.m4a', '.mp4', '.mov'], filetype
        with tempfile.TemporaryDirectory() as tempdir:
            temp_flac = Path(tempdir) / (path_audio.stem + '_temp' + '.flac')
            command = ['ffmpeg', '-i', audiofile, '-af', 'aformat=s16:16000', '-ac', '1', temp_flac] 
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            audio, sr = soundfile.read(temp_flac)
    return audio

def transcribe(model, audio_batch, step_len=None):
    device = model.device()
    if step_len is None:
        step_len = [None]*audio_batch.shape[0]
    t_audio = audio_batch.float().div_(32768.0)
    pad_len = math.ceil(len(t_audio) / HOP) * HOP - len(t_audio)
    t_audio = F.pad(t_audio, (0, pad_len)).to(device)
    frame_out, vel_out = model(t_audio, last_states=None, random_condition=False, sampling='argmax')

    outs = []
    vel_outs = []
    for n in range(len(t_audio.shape[0])): # batch size
        outs.append(frame_out[n,:step_len[n]])
        vel_outs.append(vel_out[n,:step_len[n]])
    return outs, vel_outs

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
