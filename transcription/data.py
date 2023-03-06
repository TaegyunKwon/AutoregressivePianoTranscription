"""
modified from JongWook Kim's repository
https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/dataset.py
"""
import os
from abc import abstractmethod
from pathlib import Path
import json
import csv
import numpy as np
import math
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import soundfile

from .constants import HOP, SR, MAX_MIDI, MIN_MIDI
from .midi import parse_midi, parse_pedal

def uniform_augmentation(arr, width, prob):
    mask = arr > 0
    idx = th.nonzero(mask, as_tuple=True)
    n_change = int(len(idx[0])*prob)
    perm = th.randperm(len(idx[0]))[:n_change]
    idx = [el[perm] for el in idx]
    rand_arr = th.rand((n_change,))
    out_tensor = arr.clone()
    out_tensor[idx] = out_tensor[idx]*(1+(rand_arr*2-1)*width)
    return out_tensor

def onset_time_transform(arr, std, prob, zero_prob):
    mask = arr > 0
    prob_arr = th.rand_like(arr)
    indicator = (prob_arr < prob)
    prob_arr2 = th.randn_like(arr)
    transformed = ~indicator * arr + mask*indicator*arr*(prob_arr2*std + 1)
    zero_prob_arr = th.rand_like(arr)
    zero_indicator = (zero_prob_arr < zero_prob)
    transformed = transformed * ~zero_indicator
    return transformed.to(arr.device)

def vel_transform(arr, var, prob, zero_prob):
    mask = arr > 0
    prob_arr = th.rand_like(arr)
    indicator = (prob_arr < prob)
    prob_arr2 = th.randn_like(arr)
    transformed = ~indicator * arr + mask*indicator*arr*(prob_arr2*var + 1)
    zero_prob_arr = th.rand_like(arr)
    zero_indicator = (zero_prob_arr < zero_prob)
    transformed = transformed * ~zero_indicator
    return transformed.to(arr.device)

class PianoSampleDataset(Dataset):
    def __init__(self, path, groups=None, sample_length=16000*5, seed=1, 
                 random_sample=True, transform=None, delay=1):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.random_sample = random_sample
        assert all(group in self.available_groups() for group in self.groups)
        self.sample_length = sample_length
        if self.sample_length is not None:
            assert sample_length % HOP == 0
        self.delay = delay
        self.random = np.random.RandomState(seed)
        self.transform = transform

        self.data_path = []
        self.max_last = 330

        self.file_list = dict()
        
        # outputs
        self.frame_features = ['label', 'pedal_label', 'velocity', 
                               'last_onset_time', 'last_onset_vel']
        # aggregate files in all groups
        for group in groups:
            self.file_list[group] = self.files(group)
            for input_pair in self.file_list[group]:
                self.data_path.append(input_pair)

    def __getitem__(self, index):
        '''
        audio-aligned  : - - - - - -|1 1 1 1 1 - - - - 1 1 1 1 1 - - 
                            delay-> | |
        shifted_label  : 0 0 0 0 0 0 0|1 1 1 1 1 0 0 0 0 1 1 1 1 1 0 
        shifted_pedal  : ...
        shifted_vel    : ...
        last_onset_time: 0 0 0 0 0 0 0|1 2 3 4 5 0 0 0 0 1 2 3 4 5 6 
        last_onset_vel : 0 0 0 0 0 0 0|v v v v v 0 0 0 0 V V V V V V 
        '''

        audio_path = self.data_path[index][0]
        saved_data_path = audio_path.replace('.flac', '_parsed.pt').replace('.wav', '_parsed.pt')
        data = th.load(saved_data_path)
        result = dict(path=audio_path)

        if self.sample_length is not None:  # fixed length segmentation
            audio_length = len(data['audio'])
            if self.random_sample: # select from random position
              step_begin = self.random.randint(audio_length - self.sample_length) // HOP
            else:
              step_begin = 0
            n_steps = self.sample_length // HOP
            step_end = step_begin + n_steps

            begin = step_begin * HOP
            end = begin + self.sample_length

            result['audio'] = data['audio'][begin:end]

            if step_begin > self.delay - 1:
                for el in self.frame_features:
                    result[el] = data[el][step_begin - self.delay:step_end, :]
            else:  # if no previous frames exist
                for el in self.frame_features:
                    result[el] = F.pad(data[el][step_begin:step_end, :], (0,0,self.delay,0))
            result['time'] = begin / SR 
        else: # use whole sequence at ones; padding
            audio = data['audio']
            pad_len = math.ceil(len(audio) / HOP) * HOP - len(audio)
            result['audio'] = F.pad(audio, (0, pad_len))
            for el in self.frame_features:
                result[el] = F.pad(data[el], (0,0,self.delay,0))
            
            result['time'] = data['time']

        result['audio'] = result['audio'].float().div_(32768.0)

        # make 'last onset features'
        frame_mask = result['label'] > 0
        last_onset_time = th.clamp(result['last_onset_time'], 0, 156) * frame_mask # 5sec
        last_onset_vel = th.clamp(result['last_onset_vel'], 0, 128) * frame_mask
        
        last_onset_time = last_onset_time.float()
        last_onset_vel = last_onset_vel.float()
        if self.transform:
            last_onset_time = uniform_augmentation(last_onset_time, 0.2, 0.3)
            last_onset_vel = uniform_augmentation(last_onset_vel, 0.2, 0.3)
        
        result['label'] = result['label'].long()
        result['pedal_label'] = result['pedal_label'].long()
        result['velocity'] = result['velocity'].long()
        result['last_onset_time'] = last_onset_time.div_(156)
        result['last_onset_vel'] = last_onset_vel.div_(128)

        return result

    def __len__(self):
        return len(self.data_path)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def initialize(self):
        for input_pair in tqdm(self.data_path, desc='initialize files:', ncols=100):
            self.load(*input_pair) 
        
    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Ex:
        Label          : 0 0 0 0 0|3 4 4 4 4 4 4 1 0 0  0  ..
        last_onset_time: 0 0 0 0 0|1 2 3 4 5 6 7 8 9 10 11 ..
        lase_onset_vel : 0 0 0 0 0|v v v v v v v v v v  v  .. 
        """
        saved_data_path = audio_path.replace('.flac', '_parsed.pt').replace('.wav', '_parsed.pt')
        if Path(saved_data_path).exists():
            return 

        try:
            audio, sr = soundfile.read(audio_path, dtype='int16')
        except:
            print(audio_path)
        assert sr == SR

        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=-1)

        audio = th.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP + 1

        label = th.zeros(n_steps, n_keys, dtype=th.uint8)
        velocity = th.zeros(n_steps, n_keys, dtype=th.uint8)
        last_onset_vel = th.zeros(n_steps, n_keys, dtype=th.uint8)
        last_onset_time = th.zeros(n_steps, n_keys, dtype=th.int32)

        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        last_onset_loc = -th.ones(88, dtype=th.int32)
        last_onset_vel_save = -th.ones(88, dtype=th.int32)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SR / HOP))
            onset_right = left + 1
            frame_right = int(round(offset * SR / HOP))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + 1)

            if left > n_steps:
                print(f'Warning: onset after audio ends {audio_length//SR}. {audio_path}. \
                      {onset},{offset},{note},{vel}')
                return

            # off->off :0, on -> off :1, off->onset :2, on -> on :3, on -> onset :4,
            f = int(note) - MIN_MIDI

            if last_onset_loc[f] != -1:
                last_onset_time[last_onset_loc[f]:left, f] = th.arange(1, int(left - last_onset_loc[f] + 1))
                last_onset_vel[last_onset_loc[f]:left, f] = last_onset_vel_save[f] * th.ones(left - last_onset_loc[f])
            last_onset_loc[f] = left
            last_onset_vel_save[f] = vel
            if left > 0 and label[left-1, f] <= 1:
              label[left:onset_right, f] = 2
            elif left == 0:
              label[:onset_right, f] = 2
            else:
              label[left:onset_right, f] = 4
            label[onset_right:frame_right, f] = 3
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel
        
        for f in range(88):
            if last_onset_loc[f] == -1:
                continue
            last_onset_time[last_onset_loc[f]:, f] = th.arange(1, int(n_steps - last_onset_loc[f] + 1))
            last_onset_vel[last_onset_loc[f]:, f] = last_onset_vel_save[f] * th.ones(n_steps - last_onset_loc[f])

        pedal_tsv_path = tsv_path.replace('.tsv', '_pedal.tsv')
        pedal_label = th.zeros(n_steps, 2, dtype=th.uint8)
        pedal = np.loadtxt(pedal_tsv_path, delimiter='\t', skiprows=1, ndmin=2)

        for onset, offset, pedal_type in pedal:
            left = int(round(onset * SR / HOP))
            onset_right = left + 1
            frame_right = int(round(offset * SR / HOP))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + 1)
            if pedal_type == 0.0:
                type_idx = 0
            elif pedal_type == 2.0:
                type_idx = 1
            # off->off :0, on -> off :1, off->onset :2, on -> on :3, on -> onset :4,
            if left > 0 and label[left-1, type_idx] <= 1:
              pedal_label[left:onset_right, type_idx] = 2
            elif left == 0:
              pedal_label[:onset_right, type_idx] = 2
            else:
              pedal_label[left:onset_right, type_idx] = 4
            pedal_label[onset_right:frame_right, type_idx] = 3
            pedal_label[frame_right:offset_right, type_idx] = 1

        data = dict(path=audio_path, audio=audio, label=label, pedal_label=pedal_label, velocity=velocity, 
                    last_onset_time=last_onset_time, last_onset_vel=last_onset_vel, time=0)
        th.save(data, saved_data_path)
        return 


class MAESTRO(PianoSampleDataset):
    def __init__(self, path='data/MAESTRO', json_file='maestro-v1.0.0.json', groups=None, sequence_length=None, seed=1, 
                 random_sample=True, transform=None):
        self.json_file = json_file
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, 
                         random_sample=random_sample, transform=transform)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test', 'debug']

    def files(self, group):
        metadata = json.load(open(os.path.join(self.path, self.json_file)))

        if group == 'debug':
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if
                            row['split'] == 'train'])
            files = files[:50]
        else:
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []

        first_tsv = files[0][1].replace('.midi', '.tsv').replace('.mid', '.tsv')
        if not os.path.exists(first_tsv):
          for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group, ncols=100):
              tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
              midi = parse_midi(midi_path)
              np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
              pedal = parse_pedal(midi_path)
              np.savetxt(tsv_filename.replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
              result.append((str(audio_path), str(tsv_filename)))
        else:
          # for audio_path, midi_path in files:
          for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group, ncols=100):
              tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
              result.append((str(audio_path), str(tsv_filename)))
        return result


class MAESTRO_V3(PianoSampleDataset):
    def __init__(self, path='data/maestro-v3.0.0', meta_file='maestro-v3.0.0.csv', groups=None, sequence_length=None, seed=1, 
                 random_sample=True, transform=None):
        self.meta_file = meta_file
        self.path = Path(path)
        super().__init__(self.path, groups, sequence_length, seed, random_sample, transform)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test', 'debug']

    def files(self, group):
        metadata = csv.reader(open(self.path / self.meta_file, 'r',))

        files = sorted([((self.path /row[5]).with_suffix('.flac'), self.path/row[4]) for row in metadata if
                row[2] == group])
        result = []

        for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group, ncols=100):
            tsv_filename = midi_path.with_suffix('.tsv')
            if not tsv_filename.exists():
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
                pedal = parse_pedal(midi_path)
                np.savetxt(tsv_filename.parent / (tsv_filename.stem + '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
            else:
                pass
            result.append((str(audio_path), str(tsv_filename)))
        return result
        

class MAPS(PianoSampleDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True, delay=1, audio_transforms=None, last_time_transform=None, last_vel_transform=None):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, load_mode, random_sample, delay, audio_transforms, last_time_transform, last_vel_transform)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        print(group)
        flacs = list((Path(self.path) / 'flac').glob(f'*_{group}.flac'))
        midis = [(Path(self.path) / 'midi') / el.with_suffix('.mid').name for el in flacs]
        tsvs = [(Path(self.path) / 'midi') / el.with_suffix('.tsv').name for el in flacs]

        result = []
        first_tsv = tsvs[0]
        if not os.path.exists(first_tsv):
          for audio_path, midi_path in tqdm(zip(flacs, midis), desc='Converting midi to tsv group %s' % group, ncols=100):
              tsv_filename = midi_path.with_suffix('.tsv')
              midi = parse_midi(midi_path)
              np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
              pedal = parse_pedal(midi_path)
              np.savetxt(str(tsv_filename).replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
              result.append((str(audio_path), str(tsv_filename)))
        else:
          for audio_path, midi_path in zip(flacs, midis):
              tsv_filename = midi_path.with_suffix('.tsv')
              result.append((str(audio_path), str(tsv_filename)))
        return result

class EmotionDataset(PianoSampleDataset):
    def __init__(self, path='data/Emotion', json_file='meta.json', groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True, 
                 delay=1, audio_transforms=None, last_time_transform=None, last_vel_transform=None):
        self.json_file = json_file
        super().__init__(path, groups if groups is not None else ['test'], sequence_length, seed, load_mode, random_sample,
                         delay, audio_transforms, last_time_transform, last_vel_transform)
        
    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        metadata = json.load(open(os.path.join(self.path, self.json_file)))

        if group == 'debug':
            files = sorted([(os.path.join(self.path, row['flac']),
                             os.path.join(self.path, 'midi', Path(row['flac']).with_suffix('.mid').name)) for row in metadata if
                            row['group'] == 'train'])
            files = files[:50]
        else:
            files = sorted([(os.path.join(self.path, row['flac']),
                             os.path.join(self.path, 'midi', Path(row['flac']).with_suffix('.mid').name)) for row in metadata if
                            row['group'] == group]) 

        result = []

        first_tsv = files[0][1].replace('.midi', '.tsv').replace('.mid', '.tsv')
        if not os.path.exists(first_tsv):
          for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group, ncols=100):
              tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
              midi = parse_midi(midi_path)
              np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
              pedal = parse_pedal(midi_path)
              np.savetxt(tsv_filename.replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
              result.append((audio_path, tsv_filename))
        else:
          for audio_path, midi_path in files:
              tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
              if not os.path.exists(tsv_filename):
                  continue

              result.append((audio_path, tsv_filename))
        return result


class SMD(PianoSampleDataset):
    def __init__(self, path='data/SMD', groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True, delay=1, audio_transforms=None, last_time_transform=None, last_vel_transform=None):
        super().__init__(path, groups if groups is not None else ['test'], sequence_length, seed, load_mode, random_sample, delay, audio_transforms, last_time_transform, last_vel_transform)

    @classmethod
    def available_groups(cls):
        return ['test']

    def files(self, group):
        flacs = list((Path(self.path) / 'midi_audio').glob('*flac'))
        midis = [(Path(self.path) / 'align_midi') / el.with_suffix('.mid').name for el in flacs]
        tsvs = [(Path(self.path) / 'align_midi') / el.with_suffix('.tsv').name for el in flacs]

        result = []
        first_tsv = tsvs[0]
        if not os.path.exists(first_tsv):
          for audio_path, midi_path in tqdm(zip(flacs, midis), desc='Converting midi to tsv group %s' % group, ncols=100):
              tsv_filename = midi_path.with_suffix('.tsv')
              midi = parse_midi(midi_path)
              np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
              pedal = parse_pedal(midi_path)
              np.savetxt(str(tsv_filename).replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
              result.append((str(audio_path), str(tsv_filename)))
        else:
          for audio_path, midi_path in zip(flacs, midis):
              tsv_filename = midi_path.with_suffix('.tsv')
              result.append((str(audio_path), str(tsv_filename)))
        return result

class ViennaCorpus(PianoSampleDataset):
    def __init__(self, path='data/vienna_corpus', groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True, 
                 delay=1, audio_transforms=None, last_time_transform=None, last_vel_transform=None):
        super().__init__(path, groups if groups is not None else ['test'], sequence_length, seed, load_mode, random_sample,
                         delay, audio_transforms, last_time_transform, last_vel_transform)
        
    @classmethod
    def available_groups(cls):
        return ['test']

    def files(self, group):
        midis = list(Path(self.path).glob('**/*_original.mid'))
        midis = [el for el in midis if 'average' not in el.name]
        files = sorted([(str(el.parent / el.name.replace('_original.mid', '.flac')), str(el)) for el in midis])

        result = []

        for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group, ncols=100):
            tsv_filename = midi_path.replace('.mid', '.tsv').replace('.mid', '.tsv')
            if not tsv_filename.exists():
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
                pedal = parse_pedal(midi_path)
                np.savetxt(tsv_filename.replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
                result.append((audio_path, tsv_filename))
            else:
                pass
            result.append((audio_path, tsv_filename))