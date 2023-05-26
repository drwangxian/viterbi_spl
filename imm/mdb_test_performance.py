"""
performance by original IMM, n_iters = 100

0/12 - 0.35372790681925914
1/12 - 0.34603892658680296
2/12 - 0.16106320043327918
3/12 - 0.4183634268152132
4/12 - 0.6237988884849115
5/12 - 0.5621627125547297
6/12 - 0.25837618824996106
7/12 - 0.5873391559437295
8/12 - 0.7473748754502951
9/12 - 0.23138195601618722
10/12 - 0.47492989718253437
11/12 - 0.2674426908519603
average - 0.41933331878240515


"""


DEBUG = False

import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)

import os
import librosa
import numpy as np
import mir_eval
import medleydb as mdb
from self_defined import is_vocals_m2m3_fn as is_vocals_fn
from tf_imm import IMM


def get_mdb_test_split_fn():

    val_songlist = ["BrandonWebster_DontHearAThing", "BrandonWebster_YesSirICanFly",
                    "ClaraBerryAndWooldog_AirTraffic", "ClaraBerryAndWooldog_Boys", "ClaraBerryAndWooldog_Stella",
                    "ClaraBerryAndWooldog_TheBadGuys", "ClaraBerryAndWooldog_WaltzForMyVictims",
                    "HezekiahJones_BorrowedHeart", "InvisibleFamiliars_DisturbingWildlife", "Mozart_DiesBildnis",
                    "NightPanther_Fire", "SecretMountains_HighHorse", "Snowmine_Curfews"]

    assert len(val_songlist) == 13

    return val_songlist


def validity_check_of_ref_freqs_fn(freqs):

    min_melody_freq = librosa.midi_to_hz(23.6)

    all_zeros = freqs == 0.
    all_positives = freqs > min_melody_freq
    all_valid = np.logical_or(all_zeros, all_positives)
    assert np.all(all_valid)


def gen_label_fn(track_id):

    track = mdb.MultiTrack(track_id)
    assert not track.is_instrumental
    assert track.has_melody

    m2_melody = track.melody2_annotation
    m2_melody = np.asarray(m2_melody)

    is_vocals = is_vocals_fn(track_id)
    assert len(is_vocals) == len(m2_melody)

    m2_freqs = m2_melody[:, 1]
    validity_check_of_ref_freqs_fn(m2_freqs)

    vocal_freqs = np.where(is_vocals, m2_freqs, 0.)

    return vocal_freqs


if __name__ == '__main__':

    melody_ins = IMM()
    mdb_folder = os.environ['medleydb']
    track_ids = get_mdb_test_split_fn()
    if DEBUG:
        track_ids = track_ids[:3]
    num_tracks = len(track_ids)
    for idx, track_id in enumerate(track_ids):
        mix_wav_file = os.path.join(mdb_folder, track_id, track_id + '_MIX.wav')
        energies = melody_ins.SIMM_fn(mix_wav_file)
        _max = np.max(energies)
        _min = np.min(energies)
        logging.info('{}/{} - {}, {}'.format(idx, num_tracks, _min, _max))