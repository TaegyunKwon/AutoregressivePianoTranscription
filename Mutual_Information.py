from collections import defaultdict
from pathlib import Path
import argparse
import tempfile
import shutil
import subprocess
import math
from multiprocessing import Pool
from librosa.core.convert import midi_to_hz

import torch as th
import torch.nn.functional as F
import numpy as np
import soundfile
import librosa
import os

from torch.utils.data import DataLoader
from transcription.transcribe_batch import load_model, load_dataset
from transcription.precise_evaluation import extract_notes_with_reonset
from transcription.core import models, representation, decoding
from transcription.core.constants import *
from transcription.core.midi import save_midi
from transcription.core.dataset import MAESTRO_V2
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

maestro = 'data/MAESTRO_V2/'
dataset = MAESTRO_V2(maestro, groups=['train'])
# file_root = 'runs/CompactModel_split_argmax_focalloss_0.001_211110-035101/lstm/'
file_root = Path('runs/CompactModel_argmax_focalloss_0.001_211029-122638/lstm_train/')
files = list(file_root.glob('*_o0.npy'))
files[0].name

labels = []
match_files = []

file_names = [el.name for el in files]

for el in tqdm(dataset):
    if Path(el['path']).stem + '_o0.npy' in file_names:
        labels.append(el['shifted_label'][1:])
        match_files.append(Path(file_root)/(Path(el['path']).stem + '_o0.npy'))

del(dataset)
rng = np.random.default_rng()
idx =rng.permutation(len(match_files))
n_seg = len(match_files[:50])
n_per_seg = 20000 // n_seg
rng = np.random.default_rng()
o0_cat = []
o1_cat = []
label_cat = []
for n, file_name in tqdm(enumerate(match_files[:50])):
    o0 = np.load(file_name)[:labels[n].shape[0]] 
    o1 = np.load(file_name.parent / file_name.name.replace('_o0.npy', '_o1.npy'))[:labels[n].shape[0]]
    label = labels[n] 
    idx =rng.permutation(label.shape[0])[:n_per_seg]
    o0_cat.append(o0[idx])
    o1_cat.append(o1[idx])
    label_cat.append(label[idx])
print('check0')

o0_cat = np.concatenate(o0_cat, axis=0).reshape(-1, 88*48)
o1_cat = np.concatenate(o1_cat, axis=0).reshape(-1, 88*48)
label_cat = np.concatenate(label_cat, axis=0)


def process(pitch):
    mi0 = mutual_info_classif(o0_cat, (label_cat[:,pitch]>0))
    mi1 = mutual_info_classif(o1_cat, (label_cat[:,pitch]>0))
    np.save(f'mi/mi_0_{pitch}.npy', mi0)
    np.save(f'mi/mi_1_{pitch}.npy', mi1)
    return mi0, mi1
    
print('check1')
mi = []
with Pool(processes=16) as pool:
    mi = list(tqdm(pool.imap(process, range(88)), total=88))
print('check2')

mi0, mi1 = zip(*mi)
mi_o0 = np.asarray(mi0)
mi_o1 = np.asarray(mi1)

np.save('mi_0.npy', mi_o0)
np.save('mi_1.npy', mi_o1)
