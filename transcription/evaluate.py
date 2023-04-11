import sys
from collections import defaultdict
import torch as th
import numpy as np
from scipy.stats import hmean
import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm
from mir_eval.util import midi_to_hz
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity

from .constants import HOP, SR, MIN_MIDI
from .decode import extract_notes, notes_to_frames
from .data import MAESTRO, MAESTRO_V3

#  BASE       = ['off', 'offset', 'onset', 'sustain', 'reonset']
eps = sys.float_info.epsilon
def evaluate(sample, label, sample_vel=None, vel_ref=None, band_eval=False):
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

    if band_eval:
        bands = defaultdict(list)
        band_edges = midi_to_hz(np.arange(21+22, 108, step=22))
        def get_band(p, i, type='ref'):
            for n in range(len(p)):
                if p[n] < band_edges[0]:
                    bands[f'p_{type}_0'].append(p[n])
                    bands[f'i_{type}_0'].append(i[n])
                elif p[n] < band_edges[1]:
                    bands[f'p_{type}_1'].append(p[n])
                    bands[f'i_{type}_1'].append(i[n])
                elif p[n] < band_edges[2]:
                    bands[f'p_{type}_2'].append(p[n])
                    bands[f'i_{type}_2'].append(i[n])
                else:
                    bands[f'p_{type}_3'].append(p[n])
                    bands[f'i_{type}_3'].append(i[n])
        get_band(p_ref, i_ref, type='ref')
        get_band(p_est, i_est, type='est')
                    
        for k, v in bands.items():
            bands[k] = np.asarray(v)
        for band in range(4):
            if len(bands[f'i_ref_{band}']) == 0:
                continue 
            if len(bands[f'i_est_{band}']) == 0:
                metrics[f'metric/note_band{band}/precision'].append(0.0)
                metrics[f'metric/note_band{band}/recall'].append(0.0)
                metrics[f'metric/note_band{band}/f1'].append(0.0)
                metrics[f'metric/note_band{band}/overlap'].append(0.0)
                metrics[f'metric/note_band{band}_w_offset/precision'].append(0.0)
                metrics[f'metric/note_band{band}_w_offset/recall'].append(0.0)
                metrics[f'metric/note_band{band}_w_offset/f1'].append(0.0)
                metrics[f'metric/note_band{band}_w_offset/overlap'].append(0.0)
                continue
            p, r, f, o = evaluate_notes(
                bands[f'i_ref_{band}'], bands[f'p_ref_{band}'],
                bands[f'i_est_{band}'], bands[f'p_est_{band}'], offset_ratio=None)
            metrics[f'metric/note_band{band}/precision'].append(p)
            metrics[f'metric/note_band{band}/recall'].append(r)
            metrics[f'metric/note_band{band}/f1'].append(f)
            metrics[f'metric/note_band{band}/overlap'].append(o)

            p, r, f, o = evaluate_notes(
                bands[f'i_ref_{band}'], bands[f'p_ref_{band}'],
                bands[f'i_est_{band}'], bands[f'p_est_{band}'])
            metrics[f'metric/note_band{band}_w_offset/precision'].append(p)
            metrics[f'metric/note_band{band}_w_offset/recall'].append(r)
            metrics[f'metric/note_band{band}_w_offset/f1'].append(f)
            metrics[f'metric/note_band{band}_w_offset/overlap'].append(o)
        

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

    '''
    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/frame/f1'].append(hmean(
        [frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

    for key, value in frame_metrics.items():
        metrics['metric/frame/' + key.lower().replace(' ', '_')].append(value)
    '''
        
    return metrics


def reevalute(label_path, pred_path, onset_weight=1):
    labels = th.load(label_path)
    label = labels['label']
    vel_label = labels['velocity']
    pred = np.load(pred_path)
    frame_pred = pred['pred']
    vel_pred = pred['vel']
    frame_pred[:,:,2] *= onset_weight
    sample = frame_pred.argmax(-1)
    metric = evaluate(th.from_numpy(sample), label, th.from_numpy(vel_pred), vel_label)
    return metric



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('pred_path', type=Path)
    parser.add_argument('out_name', type=Path)
    parser.add_argument('-w', '--onset_weight', type=float, default=1.0)
    
    args = parser.parse_args()
    if args.dataset == 'MAESTRO_V3':
        dataset = MAESTRO_V3(groups=['test'], sequence_length=None) 
    elif args.dataset == 'MAESTRO_V1':
        dataset = MAESTRO(groups=['test'], sequence_length=None) 
    dataset.sort_by_length()
    target_paths = []
    for path_pair in dataset.data_path:
        audio_path = path_pair[0]
        label_path = audio_path.replace('.flac', '_parsed.pt').replace('.wav', '_parsed.pt')
        pred_path = args.pred_path / Path(audio_path).with_suffix('.npz').name
        target_paths.append((label_path, pred_path))

    total_metrics = defaultdict(list)
    '''
    for pair in tqdm(target_paths):
        print(pair[1])
        metric = reevalute(pair[0], pair[1], args.onset_weight)
        for k in metric.keys():
            total_metrics[k].extend(metric[k])
    
    '''
    def my_func(pair):
        return partial(reevalute, onset_weight=args.onset_weight)(pair[0], pair[1])
    with Pool(processes=4) as pool:
        metrics = list(tqdm(
            pool.imap(
                my_func, 
                target_paths
                )))
    total_metrics = defaultdict(list)
    for n in range(len(metrics)):
        for k in metrics[0].keys():
            total_metrics[k].extend(metrics[n][k])
    
    with open(Path(args.out_name).with_suffix('.txt'), 'w') as f:
        print('test metric')
        for key, value in total_metrics.items():
            _, category, name = key.split('/')
            multiplier = 100
            if 'err' in key:
                multiplier=1
            metric_string = f'{category:>32} {name:26}: {np.mean(value)*multiplier:.3f} +- {np.std(value)*multiplier:.3f}'
            print(metric_string)
            f.write(metric_string + '\n')