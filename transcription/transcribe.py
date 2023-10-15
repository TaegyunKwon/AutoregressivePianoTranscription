from collections import defaultdict
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import subprocess
import argparse

import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import soundfile
from tqdm import tqdm
from mir_eval.util import hz_to_midi, midi_to_hz

from transcription.constants import HOP, SR, MIN_MIDI
from transcription.model import ARModel
from transcription.train import get_dataset, PadCollate
from transcription.decode import extract_notes, notes_to_frames
from transcription.midi import save_midi


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

def transcribe(model, audio_batch, save_name, step_len=None, device='cuda'):
    print('Transcription Start')
    if step_len is None:
        step_len = [None]*audio_batch.shape[0]
    t_audio = audio_batch.float()
    pad_len = math.ceil(len(t_audio) / HOP) * HOP - len(t_audio)
    t_audio = F.pad(t_audio, (0, pad_len)).to(device)
    with th.no_grad():
        frame_out, vel_out = model(t_audio, last_states=None, random_condition=False, 
                                sampling='argmax', max_step=1000)
    out = th.argmax(frame_out[0], dim=-1)
    onset_est = ((out == 2) + (out == 4))
    frame_est = ((out == 2) + (out == 3) + (out == 4))
    vel_est = th.clamp(vel_out[0]*128, min=0, max=128)
    p_est, i_est, v_est = extract_notes(onset_est, frame_est, vel_est)
    scaling = HOP / SR
    i_est = (i_est * scaling).reshape(-1, 2)

    save_midi(Path(save_name).with_suffix('.mid'), p_est+21, i_est, v_est)

    return frame_out, vel_out

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='runs/PAR_v2_230524-181231_PAR_v2_new2_herby/model_200k_0.9019.pt')
    parser.add_argument('--audio_path', type=str, required=True)
    args = parser.parse_args()

    model, config = load_model(args.model_path, 'cuda')
    print(f'load model:{args.model_path}')

    audio = load_audio(args.audio_path)
    audio_tensor = th.from_numpy(audio).unsqueeze(0)
    out, vel_out = transcribe(model, audio_tensor, 'transcribed/' + Path(args.audio_path).stem, device='cuda')