import numpy as np
import torch as th
from torch.nn import functional as F

from .representation import convert2onsets_and_frames


def random_modification(tensor, change_prob, reonset_prob=0.03, onset_prob=0.05, offset_prob=0.05, sustain_prob=0.4, off_prob=0.47):
    # TODO: fix
    mask = (tensor > 0)
    mask_near = (F.pad(mask[:, 3:], (0,0,0,3)) + F.pad(mask[:,:-3], (0,0,3,0)))>0
    idx = th.nonzero(mask_near, as_tuple=True)
    n_change = int(len(idx[0])*change_prob)
    idx = [el[:n_change] for el in idx]

    rand_arr = th.multinomial(
        th.tensor([off_prob, onset_prob, offset_prob, sustain_prob, reonset_prob]), 
        n_change, replacement=True).to(tensor.device)
    tensor[idx] = rand_arr
    return tensor

def update_context(last_onset_time, last_onset_vel, frame, vel, rep_type='base'):
    #  last_onset_time : 88
    #  last_onset_vel  : 88
    #  frame: 88
    #  vel  : 88
    
    onsets, _, frames = convert2onsets_and_frames(frame, rep_type)

    cur_onset_time = th.zeros_like(last_onset_time)
    cur_onset_vel = th.zeros_like(last_onset_vel)

    onset_pos = onsets == 1
    frame_pos = onsets != 1 and frames == 1
    empty_pos = onsets == 0 and frames == 0

    cur_onset_time = onset_pos + frame_pos*(last_onset_time+1)
    cur_onset_vel = onset_pos*vel + frame_pos*last_onset_vel
    return cur_onset_time, cur_onset_vel