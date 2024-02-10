from collections import defaultdict
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import subprocess
import argparse
import bisect

import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import librosa

import soundfile
from tqdm import tqdm
from mir_eval.util import hz_to_midi, midi_to_hz

from transcription.constants import HOP, SR, MIN_MIDI
from transcription.model import ARModel
from transcription.train import get_dataset, PadCollate
from transcription.decode import extract_notes, notes_to_frames
from transcription.midi import save_midi


def load_audio(audiofile, stop=None):
    try:
        audio, sr = soundfile.read(audiofile, stop=stop)
        if audio.shape[1] != 1:
            raise Exception
        if sr != 16000:
            raise Exception
    except:
        print(audiofile)
        path_audio = Path(audiofile)
        filetype = path_audio.suffix
        assert filetype in ['.mp3', '.ogg', '.flac', '.wav', '.m4a', '.mp4', '.mov'], filetype
        with tempfile.TemporaryDirectory() as tempdir:
            temp_flac = Path(tempdir) / (path_audio.stem + '_temp' + '.flac')
            command = ['ffmpeg', '-i', audiofile, '-af', 'aformat=s16:16000', '-ac', '1', temp_flac] 
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            audio, sr = soundfile.read(temp_flac, stop=stop)
    return audio

def transcribe(model, audio_batch, save_name, save=False, step_len=None, device='cuda', detune=0.0):
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
    if detune != 0.0:
        compensation_speed = (detune/1200 + 1)
        scaling = HOP / compensation_speed / SR 
        print(scaling)
    else:
        scaling = HOP/ SR
    i_est = (i_est * scaling).reshape(-1, 2)

    if save:
        save_midi(Path(save_name).with_suffix('.mid'), p_est+21, i_est, v_est)

    return (p_est+21, i_est, v_est), frame_out, vel_out

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



def estimate_detune(y, sr, notes):
    detunes = []
    notes = zip(*notes)
    for note in notes:
        start = note[1][0]
        end = note[1][1]
        vel = note[2]
        pitch = note[0]
        if end - start > 0.3 and vel > 60:
            if end - start > 2.0:
                end = start + 2.0  # maximum 2 seconds
            y_seg = y[int(start*sr):int(end*sr)]
            len_y = len(y_seg)
            S = librosa.stft(y=y_seg, n_fft=len(y_seg), win_length=len(y_seg), hop_length=len(y_seg), center=False)
            f_min = librosa.midi_to_hz(pitch-1)
            f_max = librosa.midi_to_hz(pitch+1)
            freqs = librosa.fft_frequencies(sr=sr,n_fft=len_y)
            freq_left = bisect.bisect_left(freqs, f_min)
            freq_right = bisect.bisect_right(freqs, f_max)
            x = np.abs(S[freq_left:freq_right+1])[:,0]
            # peaks, _ = scipy.signal.find_peaks(x, prominence=20)
            peak = np.argmax(x)
            ref_pitch = pitch
            ref = librosa.midi_to_hz(ref_pitch)

            '''
            if len(peaks) == 0:
                continue
            else:
                peak = peaks[np.argmin(peaks-ref_pitch)]
            '''
            # pitch, mag = librosa.piptrack(S=S, fmin=f_min, fmax=f_max, sr=sr, n_fft=len_y, center=False)
            # pitch, mag = librosa.piptrack(y=y_seg, sr=sr, n_fft=len_y, win_length=len_y, center=False)
            
            # p = np.max(pitch)
            if peak == len(x) - 1:
                p = (x[peak-1]*freqs[freq_left+peak-1] + 
                    x[peak]*freqs[freq_left+peak]) \
                    / np.sum(x[peak-1:peak+1])
            elif peak == 0:
                p = (x[peak]*freqs[freq_left+peak] +
                    x[peak+1]*freqs[freq_left+peak+1]) \
                    / np.sum(x[peak:peak+2])
            else:
                p = (x[peak-1]*freqs[freq_left+peak-1] + 
                    x[peak]*freqs[freq_left+peak] +
                    x[peak+1]*freqs[freq_left+peak+1]) \
                    / np.sum(x[peak-1:peak+2])
            # p = freqs[freq_left+peak]
            detune = 1200*(p-ref)/ref
            if detune > 50 or detune < -50:
                continue
            detunes.append(detune)
    return np.median(detunes)        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='runs/PAR_v2_230524-181231_PAR_v2_new2_herby/model_200k_0.9019.pt')
    parser.add_argument('--audio_path', type=str)
    parser.add_argument('--target_folder', type=str)
    parser.add_argument('--save_folder', type=Path)
    parser.add_argument('--max_len', type=int, default=100)
    args = parser.parse_args()

    assert args.audio_path is not None or args.target_folder is not None, 'Either audio_path or target_folder should be specified'
     
    if args.audio_path:
        audio_files = [Path(args.audio_path)]
        result_txt = Path(args.audio_path).parent /  'detune.txt'
        results = []
    else:
        flac_files = Path(args.target_folder).glob('**/*.flac')
        # flac_files = [flac_file for flac_file in flac_files if 'average' not in flac_file.stem]
        flac_files = [flac_file for flac_file in flac_files if 'ENST' in flac_file.stem]
        
        # audio_files = [flac_file for flac_file in flac_files 
        #       if not flac_file.with_suffix('.cue').exists() and not (flac_file.parent / (flac_file.stem + '_16k.flac')).exists()]

        # flac_files = Path(args.target_folder).glob('**/*.mp3')
        # audio_files = [flac_file for flac_file in flac_files 
        #       if 'midi' not in flac_file.stem]
        audio_files = [flac_file for flac_file in flac_files]
        result_txt = Path(args.target_folder) / 'detune.txt'
        results = []

    model, config = load_model(args.model_path, 'cuda')
    print(f'load model:{args.model_path}')

    for audio_path in tqdm(audio_files):
        # detune_path = Path(audio_path).parent / (Path(audio_path).stem + '_detune.txt')
        # if detune_path.exists():
        #     continue
        audio = load_audio(audio_path, stop=args.max_len*SR)
        audio_tensor = th.from_numpy(audio).unsqueeze(0)
        notes, frame_out, vel_out = transcribe(model, audio_tensor, Path(audio_path).parent / Path(audio_path).stem, save=False, device='cuda')
        detune = estimate_detune(audio, SR, notes)
        results.append((audio_path, detune))
        detune = detune - 4  # detune - 6 if want to adjust to +6
        

        print(f'{audio_path.stem} detune: {detune:2.1f} cents')
        # detune_path = Path(audio_path).parent / (Path(audio_path).stem + '_detune.txt')
        # with open(detune_path, 'w') as f:
        #     f.write(f'{detune:2.1f}')
        
        # transcription
        
        audio = load_audio(audio_path)
        if detune < -1 or detune > 2: 
            # ratio = (detune + 1) / 1 , compensation = 1 / speed = ratio
            compensation_speed = (detune/1200 + 1)
            audio = librosa.resample(audio, orig_sr=SR, target_sr=SR*compensation_speed)

        audio_tensor = th.from_numpy(audio).unsqueeze(0)
        if args.save_folder:
            notes, frame_out, vel_out = transcribe(model, audio_tensor, args.save_folder / Path(audio_path).stem, save=True, device='cuda', detune=detune)
        else:
            notes, frame_out, vel_out = transcribe(model, audio_tensor, Path(audio_path).parent / Path(audio_path).stem, save=True, device='cuda', detune=detune)
        
        
    if args.target_folder:
        with open(result_txt, 'w') as f:
            for result in results:
                f.write(f'{result[0]} detune: {result[1]:2.1f} cents\n')