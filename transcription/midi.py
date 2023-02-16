import multiprocessing
import sys

import mido
import numpy as np
from joblib import Parallel, delayed
from mido import Message, MidiFile, MidiTrack
from mir_eval.util import hz_to_midi
from tqdm import tqdm


def parse_midi(path, threshold=64):
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = mido.MidiFile(path)

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= threshold) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= threshold
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n is events[-1])

        note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes)


def parse_pedal(path):
    """open midi file and return np.array of (pedal_on, pedal_off) rows"""
    midi = mido.MidiFile(path)

    time = 0
    sustain = False
    sostenuto = False
    soft = False

    sustein_events = []
    sostenuto_events = []
    soft_events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change':
            if message.control == 64 and (message.value >= 64) != sustain:
                # sustain pedal state has just changed
                sustain = message.value >= 64
                event_type = 'on' if sustain else 'off'
                event = dict(index=len(sustein_events), time=time, type=event_type, note=None, velocity=0)
                sustein_events.append(event)
            elif message.control == 66 and (message.value >= 64) != sostenuto:
                # sustain pedal state has just changed
                sostenuto = message.value >= 64
                event_type = 'on' if sostenuto else 'off'
                event = dict(index=len(sostenuto_events), time=time, type=event_type, note=None, velocity=0)
                sostenuto_events.append(event)
            elif message.control == 67 and (message.value >= 64) != soft:
                # sustain pedal state has just changed
                soft = message.value >= 64
                event_type = 'on' if soft else 'off'
                event = dict(index=len(soft_events), time=time, type=event_type, note=None, velocity=0)
                soft_events.append(event)

    def _clean_out_pedal(events, type):
        pedals = []
        for i, onset in enumerate(events):
            if onset['type'] == 'off':
                continue
            if i == len(events)-1:
                # last pedal event is on. Add offset on last.
                offset = dict(time=time)
            else:
                # find the next pedal off message
                offset = next(n for n in events[i + 1:] if n['type'] == 'off' or n is events[-1])
            pedal = (onset['time'], offset['time'], type)
            pedals.append(pedal)
        return pedals

    pedals = []
    if sustein_events:
        sustein_pedals = _clean_out_pedal(sustein_events, 0)
        pedals.extend(sustein_pedals)
    if sostenuto_events:
        sostenuto_pedals = _clean_out_pedal(sostenuto_events, 1)
        pedals.extend(sostenuto_pedals)
    if soft_events:
        soft_pedals = _clean_out_pedal(soft_events, 2)
        pedals.extend(soft_pedals)
    return np.asarray(pedals)


def save_midi(path, pitches, intervals, velocities, pedal=None):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values [0 127]
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=64))
    if pedal:
        for i in range(len(pedal)):
            events.append(dict(type='pedal',control=64, time=pedal[i][0], value=pedal[i][1]))


    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        if event['type'] != 'pedal':
            velocity = int(event['velocity'])
            if velocity > 127:
                velocity = 127
            # pitch = int(round(hz_to_midi(event['pitch'])))
            pitch = event['pitch']
            try:
                track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
            except:
                print(pitch, velocity)
        else:
            try:
                track.append(Message('control_change', control=64, value=event['value'], time=current_tick - last_tick))

            except:
                print(event)
        last_tick = current_tick

    file.save(path)


def velocity_approximation(pitches, interval, pitches_ref, intervals_ref, velocities_ref):
    raise NotImplementedError


if __name__ == '__main__':
    def process(input_file, output_file):
        midi_data = parse_midi(input_file)
        np.savetxt(output_file, midi_data, '%.6f', '\t', header='onset,offset,note,velocity')


    def files():
        for input_file in tqdm(sys.argv[1:]):
            if input_file.endswith('.mid'):
                output_file = input_file[:-4] + '.tsv'
            elif input_file.endswith('.midi'):
                output_file = input_file[:-5] + '.tsv'
            else:
                print('ignoring non-MIDI file %s' % input_file, file=sys.stderr)
                continue

            yield (input_file, output_file)

    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process)(in_file, out_file) for in_file, out_file in files())
