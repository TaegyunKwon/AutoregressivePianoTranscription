from collections import defaultdict
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import subprocess

import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import soundfile
from tqdm import tqdm

from transcription.constants import HOP
from transcription.model import ARModel
from transcription.train import get_dataset, PadCollate


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

def load_model(model_path, device):
    ckp = th.load(model_path, map_location='cpu')
    config = dict()
    for k, v in ckp.items():
        if k != 'model_state_dict':
            config[k] = v
    config = SimpleNamespace(**config)
    model = ARModel(config).to(device)
    model.load_state_dict(ckp['model_state_dict'])
    model.eval()

    return model, config

def transcribe_with_lstm_out(model, config, save_folder, device='cuda'):
    test_set = get_dataset(config, ['test'], sample_len=None,
                            random_sample=False, transform=False)
    test_set.sort_by_length()
    batch_size = 1 # 6 for PAR model, 12G RAM (8 blocked by 8G shm size)
    data_loader_test = DataLoader(
        test_set, 
        batch_size=batch_size,
        num_workers=config.n_workers,
        pin_memory=False,
        collate_fn=PadCollate()
        )

    activation = []
    def get_activation(model):
        def hook(model, input, output):
            activation.append(output[0].detach().cpu().numpy())
        return hook
        
    model.lstm.register_forward_hook(get_activation(model.lstm))

    iterator = data_loader_test
    with th.no_grad():
        for batch in tqdm(iterator):
            audio = batch['audio'].to(device)
            batch_size = audio.shape[0]
            frame_out, vel_out = model(audio, last_states=None, random_condition=False, sampling='argmax')
            lstm_activation = np.concatenate(activation, axis=0)
            if config.pitchwise_lstm:
                lstm_activation = lstm_activation.reshape(-1, batch_size, 88, 48)
            for n in range(audio.shape[0]):
                step_len = batch['step_len'][n]
                lstm_out = lstm_activation[:step_len, n]
                save_path = Path(save_folder) / (Path(batch['path'][n]).stem + '.npy')
                np.save(save_path, lstm_out)
            del lstm_activation, frame_out, vel_out, lstm_out
        activation = [] 
        
if __name__ == '__main__':
    model, config = load_model('runs/PAR_v2_230420-183632_PAR_v2_cp19/model_250k_0.8988.pt', 'cuda')
    transcribe_with_lstm_out(model, config, save_folder='lstm_out/h')