import sys
from collections import defaultdict
import torch as th
import numpy as np
from scipy.stats import hmean

from mir_eval.util import midi_to_hz
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity

from .constants import HOP, SR, MIN_MIDI
from .decode import extract_notes, notes_to_frames

#  BASE       = ['off', 'offset', 'onset', 'sustain', 'reonset']
eps = sys.float_info.epsilon
def evaluate(sample, label, sample_vel=None, vel_ref=None):
    metrics = defaultdict(list)
    
    onset_est = ((sample == 2) + (sample == 4))
    frame_est = ((sample == 2) + (sample == 3) + (sample == 4))
    onset_ref = ((label == 2) + (label == 4))
    frame_ref = ((label == 2) + (label == 3) + (label == 4))

    if sample_vel is not None:
        vel_est = th.clamp(sample_vel*128, min=0, max=128)
    else:
        vel_est = th.ones_like(sample)
        vel_ref = th.ones_like(sample)
    p_est, i_est, v_est = extract_notes(onset_est, frame_est, vel_est)
    p_ref, i_ref, v_ref = extract_notes(onset_ref, frame_ref, vel_ref)

    t_est, f_est = notes_to_frames(p_est, i_est, frame_est.shape)
    t_ref, f_ref = notes_to_frames(p_ref, i_ref, frame_ref.shape)

    scaling = HOP / SR
    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [np.array([midi_to_hz(MIN_MIDI + midi)
                        for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [np.array([midi_to_hz(MIN_MIDI + midi)
                        for midi in freqs]) for freqs in f_est]

    p, r, f, o = evaluate_notes(
        i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics['metric/note/precision'].append(p)
    metrics['metric/note/recall'].append(r)
    metrics['metric/note/f1'].append(f)
    metrics['metric/note/overlap'].append(o)

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics['metric/note-with-offsets/precision'].append(p)
    metrics['metric/note-with-offsets/recall'].append(r)
    metrics['metric/note-with-offsets/f1'].append(f)
    metrics['metric/note-with-offsets/overlap'].append(o)

    p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                offset_ratio=None, velocity_tolerance=0.1)
    metrics['metric/note-with-velocity/precision'].append(p)
    metrics['metric/note-with-velocity/recall'].append(r)
    metrics['metric/note-with-velocity/f1'].append(f)
    metrics['metric/note-with-velocity/overlap'].append(o)

    p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
    metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
    metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
    metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
    metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)


    onset_loc = th.where(onset_ref)

    gt = vel_ref[onset_loc]
    est = vel_est[onset_loc]
    err = est - gt
    err = err.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    metrics['metric/onset_velocity/abs_err'].append(np.mean(np.abs(err)))
    metrics['metric/onset_velocity/rel_err'].append(np.mean(np.abs(err) / gt))

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/frame/f1'].append(hmean(
        [frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

    for key, value in frame_metrics.items():
        metrics['metric/frame/' + key.lower().replace(' ', '_')].append(value)
        
    return metrics