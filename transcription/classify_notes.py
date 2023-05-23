import argparse
import numpy as np
from .constants import HOP, MIN_MIDI, MAX_MIDI, SR
from .midi import save_midi
from .train import get_dataset, default_config
from collections import defaultdict
import mir_eval
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import match_note_offsets, precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

est_note_attr = ['match', 'offset_match', 'offset_match_long', 'velocity_match', 'time', 'length', 'length_long', 'pitch', 'velocity', 'onset_type', 'condition', 'exact_offset']
ref_note_attr = ['match', 'offset_match', 'offset_match_long', 'velocity_match', 'time', 'length', 'length_long', 'pitch', 'velocity', 'onset_type', 'ioi', 'exact_offset'] 

def match_notes(
    tsv_path,
    pred,
    vel_pred=None,
    vel_label=None,
    save_path=None, 
    sequence_length=None,
    ):
    '''
    Inference audio of a sequence into piano roll, and save and
    evaluate it with original (not-quantized) midi.
    '''
    metrics = defaultdict(list)

    ######## Label ########
    # Parse midifile to get precise (unquantized) annotation.
    # onset_ref, offset_ref, frame_ref = representation.base2onsets_and_frames(label)

    '''
    if 'MAPS' in audio_path.parts:
        tsv_path = audio_path.parent.parent / 'tsv/matched' / audio_path.with_suffix('.tsv').name
    elif audio_path.parts[0] == '..':
        tsv_path = '/'.join(audio_path.parts[1:-1]) + '/' + audio_path.with_suffix('.tsv').name
    else:
        tsv_path = audio_path.with_suffix('.tsv')
    '''
    # tsv_path = audio_path.with_suffix('.tsv')
    midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

    onsets_ref, offsets_ref, pitches_ref, vels_ref = zip(*midi)
    offset_cut = []
    reonset = [] # 0 False onset, True denote re-onset
    for n in range(len(onsets_ref)):
        m = n+1
        offset = offsets_ref[n]
        while m < len(onsets_ref) and offsets_ref[n] > onsets_ref[m]:
            # cut offset when another onset appears
            if pitches_ref[m] == pitches_ref[n]:
                offset = onsets_ref[m]
                reonset.append(True) 
                break
            m += 1
        else:
            reonset.append(False)
                
        offset_cut.append(offset)
    reonset = np.asarray(reonset)
    pitches_ref = np.asarray(pitches_ref)
    onsets_ref = np.asarray(onsets_ref)

    # calulate inter onset interval
    intervals = np.zeros((len(onsets_ref),))
    for pitch in range(MIN_MIDI, MAX_MIDI+1):
        idx = np.flatnonzero(pitches_ref==pitch)
        if len(idx) != 0:
            onset_subset = onsets_ref[idx]
            ioi = np.diff(onset_subset)
            intervals[idx[1:]] = ioi
    
    # p_ref = np.array([midi_to_hz(pitch) for pitch in pitches_ref])
    p_ref = pitches_ref
    f_ref = np.array([midi_to_hz(pitch) for pitch in pitches_ref])
    i_ref = np.array(list(zip(onsets_ref, offset_cut)))
    i_ref_long = np.array(list(zip(onsets_ref, offsets_ref)))
    v_ref = np.array(vels_ref)


    ######## Prediction ########
    p_est, i_est, i_est_long, v_est, condition_est, reonset_est = extract_notes_with_reonset(pred, vel_pred)
    # p_est = MIN_MIDI + p_est
    scaling = HOP / SR 
    i_est = i_est * scaling
    i_est_long = i_est_long * scaling
    f_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    if sequence_length:
        idx = np.where(i_ref[:,0] < sequence_length)
        i_ref = np.clip(i_ref[idx], 0, sequence_length)
        i_ref_long = np.clip(i_ref_long[idx], 0, sequence_length)
        f_ref = f_ref[idx]
        p_ref = p_ref[idx]
        v_ref = v_ref[idx]
        intervals = intervals[idx]
        reonset = reonset[idx]
    save_midi(Path(save_path).with_suffix('.mid'), p_est+21, i_est, np.rint(np.clip(v_est, 0, 127)))
    
    ######## Matching ########

    # Est_note = namedtuple('Est_note', ['match', 'offset_match', 'offset_match_long', 'velocity_match', 'length', 'pitch', 'velocity', 'onset_type', 'condition', 'exact_offset']) 
    # Ref_note = namedtuple('Ref_note', ['match', 'offset_match', 'offset_match_long', 'velocity_match', 'length', 'pitch', 'velocity', 'onset_type', 'ioi', 'exact_offset']) 
    matching = mir_eval.transcription.match_notes(i_ref, f_ref, i_est, f_est, offset_ratio=None)
    matching_off = mir_eval.transcription.match_notes(i_ref, f_ref, i_est, f_est)
    matching_off_long = mir_eval.transcription.match_notes(i_ref_long, f_ref, i_est_long, f_est)
    matching_off_exact = mir_eval.transcription.match_notes(i_ref, f_ref, i_est, f_est, offset_ratio=0.0)
    match_notes_velocity = mir_eval.transcription_velocity.match_notes(i_ref, f_ref, v_ref, i_est, f_est, v_est, offset_ratio=None)
    
    ref_notes = np.zeros((len(i_ref), len(ref_note_attr)))
    est_notes = np.zeros((len(i_est), len(est_note_attr)))

    ref_notes[:, ref_note_attr.index('pitch')] = p_ref
    ref_notes[:, ref_note_attr.index('velocity')] = v_ref
    ref_notes[:, ref_note_attr.index('time')] = i_ref[:,0]
    ref_notes[:, ref_note_attr.index('length')] = np.diff(i_ref, axis=1).squeeze()
    ref_notes[:, ref_note_attr.index('length_long')] = np.diff(i_ref_long, axis=1).squeeze()
    ref_notes[:, ref_note_attr.index('onset_type')] = np.asarray(reonset)
    ref_notes[:, ref_note_attr.index('ioi')] = intervals

    est_notes[:, est_note_attr.index('pitch')] = p_est
    est_notes[:, est_note_attr.index('velocity')] = v_est
    est_notes[:, est_note_attr.index('length')] = np.diff(i_est, axis=1).squeeze()
    est_notes[:, est_note_attr.index('time')] = i_est[:,0]
    est_notes[:, est_note_attr.index('length_long')] = np.diff(i_est_long, axis=1).squeeze()
    est_notes[:, est_note_attr.index('onset_type')] = np.asarray(reonset_est)
    est_notes[:, est_note_attr.index('condition')] = np.asarray(condition_est)
    ref_idx, est_idx = zip(*matching)
    ref_notes[ref_idx, ref_note_attr.index('match')] = 1
    est_notes[est_idx, est_note_attr.index('match')] = 1

    ref_idx, est_idx = zip(*matching_off)
    ref_notes[ref_idx, ref_note_attr.index('offset_match')] = 1
    est_notes[est_idx, est_note_attr.index('offset_match')] = 1

    ref_idx, est_idx = zip(*matching_off_long)
    ref_notes[ref_idx, ref_note_attr.index('offset_match_long')] = 1
    est_notes[est_idx, est_note_attr.index('offset_match_long')] = 1

    ref_idx, est_idx = zip(*matching_off_exact)
    ref_notes[ref_idx, ref_note_attr.index('exact_offset')] = 1
    est_notes[est_idx, est_note_attr.index('exact_offset')] = 1

    ref_idx, est_idx = zip(*match_notes_velocity)
    ref_notes[ref_idx, ref_note_attr.index('velocity_match')] = 1
    est_notes[est_idx, est_note_attr.index('velocity_match')] = 1


    # what to calculate?
    # acc per note Length
    # precise offset eval
    # how reonset was evaluated
    # how acc of (onset, condition) differ

    return ref_notes, est_notes, matching


def extract_notes_with_reonset(
    pred,
    velocity=None, 
    onset_threshold=0.5, 
    frame_threshold=0.5, 
    defalut_velocity=64):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    reonsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = ((pred == 2) + (pred == 4)).astype(int)
    frames = ((pred == 2) + (pred == 3) + (pred == 4)).astype(int)

    onset_diff = np.concatenate(
        [onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1

    if velocity is None:
        velocity = np.ones_like(onsets) * defalut_velocity

    pitches = []
    intervals = []
    intervals_long = []
    velocities = []
    conditions = []
    reonsets = []
    onset_frame, onset_pitch = onset_diff.nonzero()
    for pitch, frame in zip(onset_pitch, onset_frame):
        # frame = nonzero[0].item()
        # pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []
        offset_short = None

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break
            if (offset != onset) and onsets[offset, pitch].item():
                if offset_short == None:
                    offset_short = offset
        if offset_short == None:
            offset_short = offset

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset_short])
            intervals_long.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)
            if frame == 0:
                conditions.append(0)
            else:
                conditions.append(pred[frame-1, pitch])
            if pred[frame, pitch] == 4:
                reonsets.append(True)
            else:
                reonsets.append(False)
        else:
            raise Exception

    return np.array(pitches), np.array(intervals), np.array(intervals_long), np.array(velocities), np.array(conditions), np.array(reonsets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('eval_path', type=Path)
    args = parser.parse_args()
    
    config = SimpleNamespace(**default_config)
    config.dataset = args.dataset
    dataset = get_dataset(config, ['test'], random_sample=False, transform=False)
    for file_path in tqdm(dataset.data_path):
        tsv_path = Path(file_path[1])
        audio_path =Path(file_path[0])
        vel_path = audio_path.parent / audio_path.name.replace('.flac', '_velocity.npy')
        vel_label = np.memmap(vel_path, dtype=np.uint8, mode='c')
        vel_label = np.reshape(vel_label, (-1, 88))
        pred = np.load(args.eval_path / audio_path.with_suffix('.npz').name)
        frame = pred['pred'].argmax(-1)
        vel = np.clip(pred['vel']*128, 0, 128)
        save_path = args.eval_path / tsv_path.with_suffix('.midi').name
        ref, est, matching = match_notes(tsv_path, frame, vel, vel_label, save_path=save_path)
        np.savez(args.eval_path / tsv_path.name.replace('.tsv', '_matchnote.npz'), ref=ref, est=est, matching=matching)