

import medleydb as mdb
import numpy as np
from self_defined import is_vocals_m2m3_fn as is_vocals_fn
import librosa
import mir_eval
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from self_defined import save_np_array_to_file_fn, load_np_array_from_file_fn


N_BINS = 721
BINS_PER_OCT = 240


def gen_central_notes_fn():

    min_note = librosa.hz_to_midi(100.)
    notes = np.arange(721) * 0.05 + min_note

    return notes


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = librosa.midi_to_hz(23.6)
    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def hz_to_midi_fn(freqs):

    notes = np.zeros_like(freqs)
    positives = np.nonzero(freqs)
    notes[positives] = librosa.hz_to_midi(freqs[positives])

    return notes


def note_resampling_fn(track_id):

    track = mdb.MultiTrack(track_id)
    assert not track.is_instrumental
    assert track.has_melody

    m2_melody = track.melody2_annotation
    m2_melody = np.asarray(m2_melody)
    num_frames = len(m2_melody)

    is_vocals = is_vocals_fn(track_id)
    assert len(is_vocals) == num_frames

    m2_freqs = m2_melody[:, 1]
    validity_check_of_ref_freqs_fn(m2_freqs)

    vocal_freqs = np.where(is_vocals, m2_freqs, 0.)

    notes = hz_to_midi_fn(vocal_freqs)

    return notes


def ref_notes_quantization_fn(ref_notes):

    n_bins = N_BINS
    assert BINS_PER_OCT % 12 == 0
    bins_per_semitone = BINS_PER_OCT // 12

    t = gen_central_notes_fn()
    min_note, max_note = t[[0, -1]]
    assert min_note > 0

    _ref_notes = []
    for note in ref_notes:
        if note > 0:
            note = max(note, min_note)
            note = min(note, max_note)

        _ref_notes.append(note)
    ref_notes = _ref_notes
    ref_notes = np.asarray(ref_notes)
    ref_notes = (ref_notes - min_note) * bins_per_semitone
    ref_notes = np.round(ref_notes)
    ref_notes = ref_notes.astype(np.int32)
    ref_notes[ref_notes < 0] = n_bins

    return ref_notes


val_songlist = ["BrandonWebster_DontHearAThing", "BrandonWebster_YesSirICanFly",
                "ClaraBerryAndWooldog_AirTraffic", "ClaraBerryAndWooldog_Boys", "ClaraBerryAndWooldog_Stella",
                "ClaraBerryAndWooldog_TheBadGuys", "ClaraBerryAndWooldog_WaltzForMyVictims",
                "HezekiahJones_BorrowedHeart", "InvisibleFamiliars_DisturbingWildlife", "Mozart_DiesBildnis",
                "NightPanther_Fire", "SecretMountains_HighHorse", "Snowmine_Curfews"]
assert len(val_songlist) == 13


n_bins = N_BINS
p_steady = np.zeros([n_bins + 1], np.int64)
transition = np.zeros([n_bins + 1, n_bins + 1], np.int64)
switch = np.zeros([2, 2], np.int64)

n_recs = len(val_songlist)
for rec_idx in range(n_recs):
    ref_notes = note_resampling_fn(val_songlist[rec_idx])
    ref_notes = ref_notes_quantization_fn(ref_notes)
    unvoiced = ref_notes == n_bins
    unvoiced = unvoiced.astype(np.int32)

    for note, _note in zip(ref_notes[:-1], ref_notes[1:]):
        p_steady[note] += 1
        transition[note, _note] += 1
    p_steady[_note] += 1

    for uv, uv_ in zip(unvoiced[:-1], unvoiced[1:]):
        switch[uv, uv_] += 1
        """
        vv vu
        uv uu
        """

p_steady = p_steady / np.sum(p_steady)

plt.plot(p_steady)
plt.savefig('p_steady.png')
plt.close()

save_np_array_to_file_fn('transition_int.dat', transition, 'transition_int')
t = np.sum(transition, axis=1)
t = np.maximum(t, 1)
transition = transition / t[:, None]
plt.matshow(transition)
plt.savefig('transition.png')
plt.close()

t = np.sum(switch, axis=1)
switch = switch / t[:, None]
print(switch)

save_np_array_to_file_fn('p_steady.dat', p_steady, 'p_steady')
_name, _data = load_np_array_from_file_fn('p_steady.dat')
assert _name == 'p_steady'
assert np.array_equal(p_steady, _data)

save_np_array_to_file_fn('transition.dat', transition, 'transition')
save_np_array_to_file_fn('switch.dat', switch, 'switch')





