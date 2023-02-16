import torch as th

BASE       = ['off', 'offset', 'onset', 'sustain', 'reonset']
THREE_RE   = ['off', 'onset', 'sustain', 'reonset']
FOUR       = ['off', 'offset','onset', 'sustain']
THREE      = ['off', 'onset', 'sustain']
BINARY     = ['off', 'sustain']


def convert_representation(base_label, rep_type):
    if rep_type == 'base':
        return base_label
    elif rep_type == 'four':
        return base2four(base_label)
    elif rep_type == 'three':
        return base2three(base_label)
    elif rep_type == 'three_re':
        return base2three_re(base_label)
    elif rep_type == 'binary':
        return base2binary(base_label)


def convert_representation_np(base_label, rep_type):
    return convert_representation(
        th.from_numpy(base_label),
        rep_type
        ).numpy()


def convert2onsets_and_frames(label, rep_type):
    # label: (T,88) or (*, T, 88)
    if len(label.shape) == 2:
        label.unsqueeze(0)

    batch_size = label.shape[0]
    if rep_type in ['base', 'four']:
        return base2onsets_and_frames(label)
    elif rep_type in ['three', 'three_re']:
        onsets = (label == 1) + (label == 3)
        frames = (label == 1) + (label == 2) + (label == 3)
        padded_frame = th.zeros((batch_size, frames.shape[1] + 1, frames.shape[2]))
        padded_frame[:, 1:] = frames
        diff = padded_frame[:, 1:] - padded_frame[:, :-1]
        offsets = diff == -1
        
    elif rep_type == 'binary':
        frames = label == 1
        padded_frame = th.zeros((batch_size, frames.shape[1] + 1, frames.shape[2]))
        padded_frame[:, 1:] = frames
        diff = padded_frame[:, 1:] - padded_frame[:, :-1]
        onsets = diff == 1
        offsets = diff == -1
        return onsets, offsets, frames
    else:
        raise KeyError(f'undefined representation: {rep_type}')

    if len(label.shape) == 2:
        onsets.squeeze(0)
        offsets.squeeze(0)
        frames.squeeze(0)
    return onsets, offsets, frames


def base2binary(base_label):
    new_label = th.zeros_like(base_label)
    new_label[base_label == 4] = 1
    new_label[base_label == 3] = 1
    new_label[base_label == 2] = 1
    return new_label

def base2four(base_label):
    new_label = th.zeros_like(base_label)
    new_label[base_label == 4] = 2
    new_label[base_label == 3] = 3
    new_label[base_label == 2] = 2
    new_label[base_label == 1] = 1
    return new_label

def base2three(base_label):
    new_label = th.zeros_like(base_label)
    new_label[base_label == 4] = 1
    new_label[base_label == 2] = 1
    new_label[base_label == 3] = 2
    new_label[base_label == 1] = 0
    return new_label

def base2three_re(base_label):
    new_label = th.zeros_like(base_label)
    new_label[base_label == 4] = 3
    new_label[base_label == 2] = 1
    new_label[base_label == 3] = 2
    new_label[base_label == 1] = 0
    return new_label

def base2onsets_and_frames(base_label):
    onsets = ((base_label == 2) + (base_label == 4))
    offsets = (base_label == 1)
    frames = ((base_label == 2) + (base_label == 3) + (base_label == 4))
    return onsets, offsets, frames