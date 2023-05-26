
"""
frame_energies = np.max(frame_energies, axis=1)
frame_energies = frame_energies * (1000. / 8)
ave - 0.69017493724823  logit - 0.040005333721637726 - prob - 0.5099999904632568

thresholding
[INFO] test
[INFO] va
[INFO] 0/12 - 0.625922679901123
[INFO] 1/12 - 0.4641311466693878
[INFO] 2/12 - 0.7276590466499329
[INFO] 3/12 - 0.5827890634536743
[INFO] 4/12 - 0.7178881168365479
[INFO] 5/12 - 0.7560839056968689
[INFO] 6/12 - 0.41658875346183777
[INFO] 7/12 - 0.6424095034599304
[INFO] 8/12 - 0.8561354875564575
[INFO] 9/12 - 0.509128212928772
[INFO] 10/12 - 0.5929028987884521
[INFO] 11/12 - 0.719298243522644
[INFO] ave - 0.6342447996139526
[INFO]
oa
[INFO] 0/12 - 0.5203494429588318
[INFO] 1/12 - 0.3403981924057007
[INFO] 2/12 - 0.6098195910453796
[INFO] 3/12 - 0.4136573076248169
[INFO] 4/12 - 0.5431880950927734
[INFO] 5/12 - 0.7311373353004456
[INFO] 6/12 - 0.3930574953556061
[INFO] 7/12 - 0.34725648164749146
[INFO] 8/12 - 0.8178125023841858
[INFO] 9/12 - 0.3782571256160736
[INFO] 10/12 - 0.3522833585739136
[INFO] 11/12 - 0.6333404779434204
[INFO] ave - 0.5067130923271179
[INFO]
viterbi
[INFO] test
[INFO] va
[INFO] 0/12 - 0.594839870929718
[INFO] 1/12 - 0.7317304611206055
[INFO] 2/12 - 0.808607280254364
[INFO] 3/12 - 0.6847291588783264
[INFO] 4/12 - 0.5409546494483948
[INFO] 5/12 - 0.6663781404495239
[INFO] 6/12 - 0.3665458858013153
[INFO] 7/12 - 0.5297353267669678
[INFO] 8/12 - 0.8050892949104309
[INFO] 9/12 - 0.7089414000511169
[INFO] 10/12 - 0.5222993493080139
[INFO] 11/12 - 0.8585541248321533
[INFO] ave - 0.6515337824821472
[INFO]
oa
[INFO] 0/12 - 0.5936547517776489
[INFO] 1/12 - 0.7274859547615051
[INFO] 2/12 - 0.8024205565452576
[INFO] 3/12 - 0.665194034576416
[INFO] 4/12 - 0.5188542008399963
[INFO] 5/12 - 0.6663272380828857
[INFO] 6/12 - 0.36639004945755005
[INFO] 7/12 - 0.42072805762290955
[INFO] 8/12 - 0.802483320236206
[INFO] 9/12 - 0.7046382427215576
[INFO] 10/12 - 0.4802710711956024
[INFO] 11/12 - 0.8530552983283997
[INFO] ave - 0.6334585547447205

Process finished with exit code 0

"""
import os.path
import time

import librosa

DEBUG = False
THRESHOLD = 2.442347

import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import numpy as np
import tensorflow as tf
from self_defined import is_vocals_m2m3_fn
import scipy.special
from tf_imm import IMM
import medleydb as mdb
from self_defined import load_np_array_from_file_fn


if DEBUG:
    for name in logging.root.manager.loggerDict:
        if name.startswith('numba'):
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING)

        if name.startswith('matplotlib'):
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING)


class Config:

    def __init__(self):

        self.debug = DEBUG
        self.tvt_split_dict = Config.get_dataset_split_fn()
        if self.debug:
            for k in self.tvt_split_dict:
                self.tvt_split_dict[k] = self.tvt_split_dict[k][:3]

        self.model_names = list(self.tvt_split_dict.keys())

        self.track_ids = self.tvt_split_dict['validation']

    @staticmethod
    def get_dataset_split_fn():

        train_songlist = ["AimeeNorwich_Child", "AlexanderRoss_GoodbyeBolero", "AlexanderRoss_VelvetCurtain",
                          "AvaLuna_Waterduct", "BigTroubles_Phantom", "DreamersOfTheGhetto_HeavyLove",
                          "FacesOnFilm_WaitingForGa", "FamilyBand_Again", "Handel_TornamiAVagheggiar",
                          "HeladoNegro_MitadDelMundo", "HopAlong_SisterCities", "LizNelson_Coldwar",
                          "LizNelson_ImComingHome", "LizNelson_Rainfall", "Meaxic_TakeAStep", "Meaxic_YouListen",
                          "MusicDelta_80sRock", "MusicDelta_Beatles", "MusicDelta_Britpop", "MusicDelta_Country1",
                          "MusicDelta_Country2", "MusicDelta_Disco", "MusicDelta_Grunge", "MusicDelta_Hendrix",
                          "MusicDelta_Punk", "MusicDelta_Reggae", "MusicDelta_Rock", "MusicDelta_Rockabilly",
                          "PurlingHiss_Lolita", "StevenClark_Bounty", "SweetLights_YouLetMeDown",
                          "TheDistricts_Vermont", "TheScarletBrand_LesFleursDuMal", "TheSoSoGlos_Emergency",
                          "Wolf_DieBekherte"]
        val_songlist = ["BrandonWebster_DontHearAThing", "BrandonWebster_YesSirICanFly",
                        "ClaraBerryAndWooldog_AirTraffic", "ClaraBerryAndWooldog_Boys", "ClaraBerryAndWooldog_Stella",
                        "ClaraBerryAndWooldog_TheBadGuys", "ClaraBerryAndWooldog_WaltzForMyVictims",
                        "HezekiahJones_BorrowedHeart", "InvisibleFamiliars_DisturbingWildlife", "Mozart_DiesBildnis",
                        "NightPanther_Fire", "SecretMountains_HighHorse", "Snowmine_Curfews"]
        test_songlist = ["AClassicEducation_NightOwl", "Auctioneer_OurFutureFaces", "CelestialShore_DieForUs",
                         "Creepoid_OldTree", "Debussy_LenfantProdigue", "MatthewEntwistle_DontYouEver",
                         "MatthewEntwistle_Lontano", "Mozart_BesterJungling", "MusicDelta_Gospel",
                         "PortStWillow_StayEven", "Schubert_Erstarrung", "StrandOfOaks_Spacestation"]

        assert len(train_songlist) == 35
        assert len(val_songlist) == 13
        assert len(test_songlist) == 12

        return dict(
            training=train_songlist,
            validation=val_songlist,
            test=test_songlist
        )


# used to determine best voicing threshold
class ValidationVoicingAccuracy:

    def __init__(self, config):


        self.config = config
        self.num_recs = len(config.track_ids)

        t = np.arange(.01, 1., .01, dtype=np.float64)
        t = np.log(t / (1. - t))
        t = t.astype(np.float32)
        self.voicing_thresholds = t
        self.num_voicing_thresholds = len(self.voicing_thresholds)
        self.var_dict = self.define_tf_variables_fn()

    def define_tf_variables_fn(self):

        num_recs = self.num_recs
        num_ths = self.num_voicing_thresholds

        with tf.name_scope('validation'):
            with tf.name_scope('statistics'):

                all_defined_vars_updated = dict()

                def gen_tf_var(name, shape):

                    assert name
                    assert shape == [num_recs] or shape == [num_recs, num_ths]

                    with tf.device('/cpu:0'):
                        var = tf.Variable(
                            initial_value=tf.zeros(shape, dtype=tf.int64),
                            trainable=False,
                            name=name
                        )
                    var_ref = var.ref()
                    assert var_ref not in all_defined_vars_updated
                    all_defined_vars_updated[var_ref] = False

                    return var

                melody_var_dict = dict(
                    gt=dict(
                        voiced=gen_tf_var('voiced', [num_recs]),
                        unvoiced=gen_tf_var('unvoiced', [num_recs])
                    ),
                    voicing=dict(
                        correct_voiced=gen_tf_var('correct_voiced', [num_recs, num_ths]),
                        incorrect_voiced=gen_tf_var('incorrect_voiced', [num_recs, num_ths]),
                        correct_unvoiced=gen_tf_var('correct_unvoiced', [num_recs, num_ths])
                    )
                )



                return dict(
                    melody=melody_var_dict,
                    all_updated=all_defined_vars_updated
                )

    def update_states_fn(self, rec_idx, frame_energies):

        assert frame_energies.shape[0] == 721
        frame_energies = np.transpose(frame_energies)

        track_ids = self.config.track_ids

        ref_voiced = is_vocals_m2m3_fn(track_ids[rec_idx])
        n_frames = len(ref_voiced)
        _n_frames = len(frame_energies)
        t = n_frames - _n_frames
        assert t >= 0
        assert t <= 1
        if t == 1:
            frame_energies = np.pad(frame_energies, [[0, 1], [0, 0]])

        frame_energies = np.max(frame_energies, axis=1)
        frame_energies = np.maximum(frame_energies, 1e-11)
        frame_energies = np.log10(frame_energies) + 6

        self.tf_update_states_fn(
            rec_idx=rec_idx,
            ref_voiced=ref_voiced,
            max_frame_energies=frame_energies
        )

    def update_melody_var_fn(self, rec_idx, l1, l2, value):

        assert rec_idx is not None
        assert l1 is not None
        assert l2 is not None
        assert value is not None


        var_dict = self.var_dict['melody']
        all_updated = self.var_dict['all_updated']


        var = var_dict[l1][l2]

        var_ref = var.ref()
        assert not all_updated[var_ref]

        with tf.device(var.device):
            value = tf.identity(value)
        var.scatter_update(tf.IndexedSlices(values=value, indices=rec_idx))
        all_updated[var_ref] = True

    @tf.function(
        input_signature=[
            tf.TensorSpec([], tf.int32, name='rec_idx'),
            tf.TensorSpec([None], tf.bool, name='ref_voiced'),
            tf.TensorSpec([None], name='max_frame_energies')
        ]
    )
    def tf_update_states_fn(self, rec_idx, ref_voiced, max_frame_energies):

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        ref_voiced = tf.convert_to_tensor(ref_voiced, tf.bool)
        max_frame_energies = tf.convert_to_tensor(max_frame_energies, tf.float32)
        max_frame_energies.set_shape([None])

        logit_thresholds = tf.convert_to_tensor(self.voicing_thresholds)
        num_thresholds = self.num_voicing_thresholds

        def count_nonzero_for_multi_ths_fn(inputs):

            inputs.set_shape([None, num_thresholds])
            outputs = tf.math.count_nonzero(inputs, axis=0, dtype=tf.int32)
            outputs = tf.cast(outputs, tf.int64)

            return outputs

        n_ref_voiced = tf.logical_not(ref_voiced)

        est_voiced = max_frame_energies[:, None] > logit_thresholds[None, :]
        est_voiced.set_shape([None, num_thresholds])
        n_est_voiced = tf.logical_not(est_voiced)

        gt_voiced = tf.math.count_nonzero(ref_voiced, dtype=tf.int64)
        gt_unvoiced = tf.size(ref_voiced, out_type=tf.int64) - gt_voiced
        correct_voiced_frames = count_nonzero_for_multi_ths_fn(ref_voiced[:, None] & est_voiced)
        correct_voiced_frames.set_shape([num_thresholds])
        incorrect_voiced_frames = count_nonzero_for_multi_ths_fn(n_ref_voiced[:, None] & est_voiced)
        correct_unvoiced_frames = count_nonzero_for_multi_ths_fn(n_ref_voiced[:, None] & n_est_voiced)

        self.update_melody_var_fn(rec_idx, 'gt', 'voiced', gt_voiced)
        self.update_melody_var_fn(rec_idx, 'gt', 'unvoiced', gt_unvoiced)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_unvoiced', correct_unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'incorrect_voiced', incorrect_voiced_frames)

        assert all(self.var_dict['all_updated'].values())

    @staticmethod
    def to_f8_divide_and_to_f4_fn(numerator, denominator):

        numerator = tf.cast(numerator, tf.float64)
        denominator = tf.cast(denominator, tf.float64)
        numerator = numerator / tf.maximum(denominator, 1e-7)
        numerator = tf.cast(numerator, tf.float32)

        return numerator

    def results(self):

        melody_dict = self.var_dict['melody']
        f8f4div = ValidationVoicingAccuracy.to_f8_divide_and_to_f4_fn
        num_recs = self.num_recs
        num_ths = self.num_voicing_thresholds

        num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        va = f8f4div(
            numerator=melody_dict['voicing']['correct_voiced'] + melody_dict['voicing']['correct_unvoiced'],
            denominator=num_frames_vector[:, None]
        )
        va.set_shape([num_recs, num_ths])
        ave_va = tf.reduce_mean(va, axis=0)
        best_th_idx = tf.argmax(ave_va, output_type=tf.int32).numpy()
        best_th_logit = self.voicing_thresholds[best_th_idx]
        best_th_prob = scipy.special.expit(best_th_logit)
        va = va[:, best_th_idx]
        va = va.numpy()
        np_ave_va = np.mean(va)
        logging.info('best threshold - logit - {} - prob - {}'.format(best_th_logit, best_th_prob))

        logging.info('va')
        for rec_idx in range(num_recs):
            logging.info('{} - {}'.format(rec_idx, va[rec_idx]))
        logging.info('ave - {}'.format(np_ave_va))


# used to evaluate validation performance under hard thresholding
class HardThresholdingInferencePerformance:

    def __init__(self, config, model_name):

        self.config = config
        assert model_name in config.model_names
        self.model_name = model_name
        self.track_ids = config.tvt_split_dict[model_name]
        self.num_recs = len(self.track_ids)

        self.voicing_threshold = THRESHOLD

        self.var_dict = self.define_tf_variables_fn()

    def update_melody_var_fn(self, rec_idx, l1, l2, value):

        assert rec_idx is not None
        assert l1 is not None
        assert l2 is not None
        assert value is not None

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['melody'][l1][l2]

        var_ref = var.ref()
        assert not all_updated[var_ref]

        with tf.device(var.device):
            value = tf.identity(value)
        var.scatter_update(tf.IndexedSlices(values=value, indices=rec_idx))
        all_updated[var_ref] = True

    @staticmethod
    def est_notes_fn(logits_argmax):

        min_note = librosa.hz_to_midi(100.)
        notes = np.arange(721) * 0.05 + min_note
        notes = notes.astype(np.float32)
        notes = tf.gather(notes, indices=logits_argmax)

        return notes

    @staticmethod
    def validity_check_of_ref_freqs_fn(freqs):

        min_melody_freq = librosa.midi_to_hz(23.6)
        all_zeros = freqs == 0.
        all_positives = freqs > min_melody_freq
        all_valid = np.logical_or(all_zeros, all_positives)
        assert np.all(all_valid)

    @staticmethod
    def hz_to_midi_fn(freqs):

        notes = np.zeros_like(freqs)
        positives = np.nonzero(freqs)
        notes[positives] = librosa.hz_to_midi(freqs[positives])

        return notes

    @staticmethod
    def gen_label_fn(track_id):

        track = mdb.MultiTrack(track_id)
        assert not track.is_instrumental
        assert track.has_melody

        m2_melody = track.melody2_annotation
        m2_melody = np.asarray(m2_melody)

        is_vocals = is_vocals_m2m3_fn(track_id)
        assert len(is_vocals) == len(m2_melody)

        m2_freqs = m2_melody[:, 1]
        HardThresholdingInferencePerformance.validity_check_of_ref_freqs_fn(m2_freqs)

        vocal_freqs = np.where(is_vocals, m2_freqs, 0.)
        notes = HardThresholdingInferencePerformance.hz_to_midi_fn(vocal_freqs)

        result = dict(notes=notes, original=dict(times=m2_melody[:, 0], freqs=vocal_freqs))

        return result

    def define_tf_variables_fn(self):

        model_name = self.model_name
        num_recs = self.num_recs

        with tf.name_scope(self.model_name):
            with tf.name_scope('statistics'):

                all_defined_vars_updated = dict()

                def gen_tf_var(name):

                    assert name

                    with tf.device('/cpu:0'):
                        var = tf.Variable(
                            initial_value=tf.zeros([num_recs], dtype=tf.int64),
                            trainable=False,
                            name=name
                        )
                    var_ref = var.ref()
                    assert var_ref not in all_defined_vars_updated
                    all_defined_vars_updated[var_ref] = False

                    return var

                melody_var_dict = dict(
                    gt=dict(
                        voiced=gen_tf_var('voiced'),
                        unvoiced=gen_tf_var('unvoiced')
                    ),
                    voicing=dict(
                        correct_voiced=gen_tf_var('correct_voiced'),
                        incorrect_voiced=gen_tf_var('incorrect_voiced'),
                        correct_unvoiced=gen_tf_var('correct_unvoiced')
                    ),
                    correct_pitches=dict(
                        wide=gen_tf_var('correct_pitches_wide'),
                        strict=gen_tf_var('correct_pitches_strict')
                    ),
                    correct_chromas=dict(
                        wide=gen_tf_var('correct_chromas_wide'),
                        strict=gen_tf_var('correct_chromas_strict')
                    )
                )

                return dict(
                    melody=melody_var_dict,
                    all_updated=all_defined_vars_updated
                )

    @staticmethod
    def count_nonzero_fn(inputs):

        outputs = inputs
        outputs = tf.math.count_nonzero(outputs, dtype=tf.int32)
        outputs = tf.cast(outputs, tf.int64)

        return outputs

    @tf.function(
        input_signature=[
            tf.TensorSpec([], tf.int32, name='rec_idx'),
            tf.TensorSpec([None], name='ref_notes'),
            tf.TensorSpec([721, None], name='logits')
        ]
    )
    def update_states_tf_fn(self, rec_idx, ref_notes, logits):

        voicing_threshold = self.voicing_threshold
        count_nonzero_fn = HardThresholdingInferencePerformance.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        logits.set_shape([721, None])
        logits = tf.transpose(logits)

        t = tf.size(ref_notes) - tf.shape(logits)[0]
        tf.debugging.assert_greater_equal(t, 0)
        tf.debugging.assert_less_equal(t, 1)
        logits = tf.cond(t == 1, lambda: tf.pad(logits, [[0, 1], [0, 0]]), lambda:logits)
        logits_argmax = tf.argmax(logits, axis=1, output_type=tf.int32)
        logits = tf.gather(logits, axis=1, batch_dims=1, indices=logits_argmax)
        logits = tf.maximum(logits, 1e-11)
        logits = tf.math.log(logits) / tf.math.log(10.) + 6.

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_voicing = logits > voicing_threshold
        n_est_voicing = tf.logical_not(est_voicing)

        est_notes = HardThresholdingInferencePerformance.est_notes_fn(logits_argmax)

        est_ref_note_diffs = tf.abs(est_notes - ref_notes)
        est_ref_note_diffs.set_shape([None])

        voiced_frames = count_nonzero_fn(ref_voicing)
        unvoiced_frames = tf.size(ref_voicing, tf.int64) - voiced_frames
        correct_voiced_frames = tf.logical_and(ref_voicing, est_voicing)
        correct_voiced_frames = count_nonzero_fn(correct_voiced_frames)
        incorrect_voiced_frames = tf.logical_and(n_ref_voicing, est_voicing)
        incorrect_voiced_frames = count_nonzero_fn(incorrect_voiced_frames)
        correct_unvoiced_frames = tf.logical_and(n_ref_voicing, n_est_voicing)
        correct_unvoiced_frames = count_nonzero_fn(correct_unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'gt', 'voiced', voiced_frames)
        self.update_melody_var_fn(rec_idx,  'gt', 'unvoiced', unvoiced_frames)
        self.update_melody_var_fn(rec_idx,  'voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn(rec_idx,  'voicing', 'incorrect_voiced', incorrect_voiced_frames)
        self.update_melody_var_fn(rec_idx,  'voicing', 'correct_unvoiced', correct_unvoiced_frames)

        correct_pitches_wide = est_ref_note_diffs < .5
        correct_pitches_wide = tf.logical_and(ref_voicing, correct_pitches_wide)
        correct_pitches_strict = tf.logical_and(est_voicing, correct_pitches_wide)
        correct_pitches_wide = count_nonzero_fn(correct_pitches_wide)
        correct_pitches_strict = count_nonzero_fn(correct_pitches_strict)
        self.update_melody_var_fn(rec_idx,  'correct_pitches', 'wide', correct_pitches_wide)
        self.update_melody_var_fn(rec_idx,  'correct_pitches', 'strict', correct_pitches_strict)

        correct_chromas_wide = est_ref_note_diffs
        octave = HardThresholdingInferencePerformance.octave(correct_chromas_wide)
        correct_chromas_wide = tf.abs(correct_chromas_wide - octave) < .5
        correct_chromas_wide = tf.logical_and(ref_voicing, correct_chromas_wide)
        correct_chromas_strict = tf.logical_and(est_voicing, correct_chromas_wide)
        correct_chromas_wide = count_nonzero_fn(correct_chromas_wide)
        correct_chromas_strict = count_nonzero_fn(correct_chromas_strict)
        self.update_melody_var_fn(rec_idx,  'correct_chromas', 'wide', correct_chromas_wide)
        self.update_melody_var_fn(rec_idx,  'correct_chromas', 'strict', correct_chromas_strict)

        assert all(self.var_dict['all_updated'].values())

        est_notes_with_voicing_info = tf.where(est_voicing, est_notes, -est_notes)

        return est_notes_with_voicing_info

    def update_states(self, rec_idx, logits):

        num_recs = self.num_recs
        assert rec_idx < num_recs

        track_ids = self.track_ids
        track_id = track_ids[rec_idx]
        ref_notes_dict = HardThresholdingInferencePerformance.gen_label_fn(track_id)
        self.update_states_tf_fn(
            rec_idx=rec_idx,
            ref_notes=ref_notes_dict['notes'],
            logits=logits
        )

    def results(self):

        num_recs = self.num_recs
        melody_dict = self.var_dict['melody']

        f8f4div = ValidationVoicingAccuracy.to_f8_divide_and_to_f4_fn

        correct_frames = melody_dict['correct_pitches']['strict'] + melody_dict['voicing']['correct_unvoiced']
        num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        m_oa = f8f4div(correct_frames, num_frames_vector)
        m_oa.set_shape([num_recs])

        m_vrr = f8f4div(melody_dict['voicing']['correct_voiced'], melody_dict['gt']['voiced'])
        m_vfa = f8f4div(melody_dict['voicing']['incorrect_voiced'], melody_dict['gt']['unvoiced'])
        m_va = f8f4div(
            melody_dict['voicing']['correct_voiced'] + melody_dict['voicing']['correct_unvoiced'],
            num_frames_vector
        )
        m_rpa_strict = f8f4div(
            melody_dict['correct_pitches']['strict'], melody_dict['gt']['voiced']
        )
        m_rpa_wide = f8f4div(
            melody_dict['correct_pitches']['wide'], melody_dict['gt']['voiced']
        )
        m_rca_strict = f8f4div(
            melody_dict['correct_chromas']['strict'], melody_dict['gt']['voiced']
        )
        m_rca_wide = f8f4div(
            melody_dict['correct_chromas']['wide'], melody_dict['gt']['voiced']
        )

        ave_oa = tf.reduce_mean(m_oa)
        m_oa = tf.pad(m_oa, [[0, 1]], mode='constant', constant_values=ave_oa)
        m_oa = m_oa.numpy()

        ave_va = tf.reduce_mean(m_va)
        m_va = tf.pad(m_va, [[0, 1]], mode='constant', constant_values=ave_va)
        m_va = m_va.numpy()

        logging.info(self.model_name)
        logging.info('va')
        for rec_idx in range(num_recs):
            logging.info('{}/{} - {}'.format(rec_idx, num_recs, m_va[rec_idx]))
        logging.info('ave - {}'.format(m_va[-1]))

        logging.info('\noa')
        for rec_idx in range(num_recs):
            logging.info('{}/{} - {}'.format(rec_idx, num_recs, m_oa[rec_idx]))
        logging.info('ave - {}'.format(m_oa[-1]))

    @staticmethod
    def octave(distance):

        distance = tf.floor(distance / 12. + .5) * 12.

        return distance


class Viterbi:

    def __init__(self):

        """
        :param voicing_threshold: a probability threshold
        """

        self.num_freq_bins = 721
        self.h = 256
        self.sr = 44100
        self.single_side_peak_width = 20

        self.threshold = THRESHOLD
        self.log_transition_matrix_T = self.transition_matrix_fn()

        self.log_ini_probs = self.init_probs_fn()

    @staticmethod
    def expit(s):

        if s > 0:
            p = 1. / (1. + np.exp(-s))
        else:
            p = np.exp(s)
            p = p / (1. + p)

        return p

    def find_peaks_all_at_once_np_fn(self, frames_logits):

        n_bins = self.num_freq_bins
        spw = self.single_side_peak_width

        assert frames_logits.ndim == 2
        assert frames_logits.shape[1] == n_bins
        n_frames = len(frames_logits)
        frames_logits = np.pad(frames_logits, [(0, 0), (spw, spw)], mode='reflect')
        w = 2 * spw + 1
        frames_are_peaks = np.zeros([n_frames, n_bins], np.bool_)
        spw = np.asarray(spw, np.int64)
        for bin_idx in range(n_bins):
            peak_idx = np.argmax(frames_logits[:, bin_idx:bin_idx + w], axis=1)
            are_peaks = peak_idx == spw
            frames_are_peaks[:, bin_idx] = are_peaks

        return frames_are_peaks

    def observation_probs_fn(self, logits):

        assert isinstance(logits, np.ndarray)
        assert logits.dtype == np.float32
        n_frames, n_freq_bins = logits.shape
        assert n_freq_bins == self.num_freq_bins
        frames_logits = logits

        threshold = self.threshold

        p = 0.8
        offset = np.log(p / (1. - p))
        scale = 2.

        melodies_frames = np.zeros([n_freq_bins + 1, n_frames], np.float32, order='F')
        frames_logits_are_peaks = self.find_peaks_all_at_once_np_fn(frames_logits)

        for frame_idx, (logits, is_peak) in enumerate(zip(frames_logits, frames_logits_are_peaks)):

            peak_indices = np.where(is_peak)[0]
            n_peaks = len(peak_indices)
            if n_peaks == 0:
                melodies_frames[-1, frame_idx] = 1
                continue

            peak_logits = logits[peak_indices]
            global_peak_local_idx = np.argmax(peak_logits)
            global_peak_logit = peak_logits[global_peak_local_idx]

            if global_peak_logit >= threshold:
                scaled_global_peak_logit = scale * (global_peak_logit - threshold) + offset
            else:
                scaled_global_peak_logit = scale * (global_peak_logit - threshold) - offset
            p_voiced = Viterbi.expit(scaled_global_peak_logit)

            peak_logits -= global_peak_logit
            np.exp(peak_logits, out=peak_logits)
            t = p_voiced / np.sum(peak_logits)
            np.multiply(peak_logits, t, out=peak_logits)
            melodies_frames[peak_indices, frame_idx] = peak_logits
            melodies_frames[-1, frame_idx] = 1. - p_voiced

        t = np.sum(melodies_frames, axis=0)
        assert np.all(np.isclose(t, 1))

        return melodies_frames

    def init_probs_fn(self):

        U = self.num_freq_bins

        file_name = 'viterbi_init_probs.dat'
        _name, probs = load_np_array_from_file_fn(file_name)
        assert _name == file_name[:-4]
        assert probs.shape == (U + 1,)
        assert np.isclose(np.sum(probs), 1)

        tiny = np.finfo(np.float32).tiny
        t = np.log(probs + tiny)
        assert not np.any(np.isneginf(t))
        t = np.require(t, np.float32)
        t.flags['WRITEABLE'] = False

        return t

    def transition_matrix_fn(self):

        U = self.num_freq_bins

        name, transition_matrix = load_np_array_from_file_fn('viterbi_transition_matrix.dat')
        assert name == 'viterbi_transition_matrix'
        assert transition_matrix.shape == (U + 1, U + 1)
        t = np.sum(transition_matrix, axis=1)
        assert np.all(np.isclose(t, 1))

        tiny = np.finfo(np.float32).tiny
        t = np.log(transition_matrix + tiny)
        assert not np.any(np.isneginf(t))
        t = t.T
        t = np.require(t, np.float32, ['C'])
        t.flags['WRITEABLE'] = False

        return t

    def __call__(self, logits):

        """
        :param logits: n_frames * n_bins
        """

        n_bins = self.num_freq_bins

        observation_probs = self.observation_probs_fn(logits)
        bins = self.viterbi_librosa_fn(observation_probs)
        voiced = bins < n_bins
        bins = np.minimum(bins, n_bins - 1)

        return voiced, bins

    def viterbi_librosa_fn(self, probs_st):

        """
        np version is faster than tf version

        """
        S = self.num_freq_bins + 1
        B = self.log_transition_matrix_T
        prob_init = self.log_ini_probs

        assert probs_st.shape[0] == S
        assert probs_st.dtype == np.float32
        assert probs_st.flags['F_CONTIGUOUS'] == True
        tinyp = np.finfo(np.float32).tiny
        np.add(probs_st, tinyp, out=probs_st)
        np.log(probs_st, out=probs_st)
        probs = np.require(probs_st.T, np.float32, ['C'])
        T = probs.shape[0]

        T1 = np.empty([T, S], np.float32)
        T2 = np.empty([T, S], np.int64)

        T1[0] = prob_init + probs[0]

        Bt = np.empty([S, S], np.float32)
        for t in range(1, T):
            np.add(T1[t - 1], B, out=Bt)
            np.argmax(Bt, axis=1, out=T2[t])
            np.add(np.take_along_axis(Bt, indices=T2[t][:, None], axis=1)[:, 0], probs[t], out=T1[t])

        states = np.empty([T], np.int64)
        s = np.argmax(T1[-1])
        states[-1] = s
        for t in range(T - 2, -1, -1):
            s = T2[t + 1, s]
            states[t] = s

        return states


# used to evaluate viterbi-decoding performance under shaun's method
class ViterbiInferencePerformance(HardThresholdingInferencePerformance):

    viterbi_ins = Viterbi()

    def __init__(self, config, model_name):

        super(ViterbiInferencePerformance, self).__init__(config=config, model_name=model_name)

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.int32, name='rec_idx'),
        tf.TensorSpec([None], name='ref_notes'),
        tf.TensorSpec([None], tf.bool, name='viterbi_est_voicing'),
        tf.TensorSpec([None], tf.int32, name='viterbi_est_bins')
    ], autograph=False)
    def update_states_tf_fn(self, rec_idx, ref_notes, viterbi_est_voicing, viterbi_est_bins):

        count_nonzero_fn = ViterbiInferencePerformance.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        rec_idx.set_shape([])

        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        ref_notes.set_shape([None])

        viterbi_est_voicing = tf.convert_to_tensor(viterbi_est_voicing, tf.bool)
        viterbi_est_voicing.set_shape([None])

        viterbi_est_bins = tf.convert_to_tensor(viterbi_est_bins, tf.int32)
        viterbi_est_bins.set_shape([None])

        n_frames = tf.size(ref_notes)
        _n_frames = tf.size(viterbi_est_voicing)
        t = n_frames - _n_frames
        tf.debugging.assert_greater_equal(t, 0)
        tf.debugging.assert_less_equal(t, 1)
        viterbi_est_voicing = tf.cond(t == 1, lambda: tf.pad(viterbi_est_voicing, [[0, 1]]), lambda:viterbi_est_voicing)
        viterbi_est_bins = tf.cond(t == 1,
                                   lambda: tf.pad(viterbi_est_bins, [[0, 1]]),
                                   lambda: viterbi_est_bins
                                   )

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_notes = HardThresholdingInferencePerformance.est_notes_fn(viterbi_est_bins)
        est_voicing = viterbi_est_voicing
        n_est_voicing = tf.logical_not(est_voicing)

        est_ref_note_diffs = tf.abs(est_notes - ref_notes)
        est_ref_note_diffs.set_shape([None])

        voiced_frames = count_nonzero_fn(ref_voicing)
        unvoiced_frames = tf.size(ref_voicing, tf.int64) - voiced_frames
        ref_voicing_logical_and_est_voicing = ref_voicing & est_voicing
        correct_voiced_frames = count_nonzero_fn(ref_voicing_logical_and_est_voicing)
        incorrect_voiced_frames = count_nonzero_fn(n_ref_voicing & est_voicing)
        correct_unvoiced_frames = count_nonzero_fn(n_ref_voicing & n_est_voicing)
        self.update_melody_var_fn(rec_idx, 'gt', 'voiced', voiced_frames)
        self.update_melody_var_fn(rec_idx, 'gt', 'unvoiced', unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'incorrect_voiced', incorrect_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_unvoiced', correct_unvoiced_frames)

        correct_pitches_wide = est_ref_note_diffs < .5
        correct_pitches_wide = tf.logical_and(ref_voicing, correct_pitches_wide)
        correct_pitches_strict = tf.logical_and(est_voicing, correct_pitches_wide)
        correct_pitches_wide = count_nonzero_fn(correct_pitches_wide)
        correct_pitches_strict = count_nonzero_fn(correct_pitches_strict)
        self.update_melody_var_fn(rec_idx,  'correct_pitches', 'wide', correct_pitches_wide)
        self.update_melody_var_fn(rec_idx,  'correct_pitches', 'strict', correct_pitches_strict)

        correct_chromas_wide = est_ref_note_diffs
        octave = HardThresholdingInferencePerformance.octave(correct_chromas_wide)
        correct_chromas_wide = tf.abs(correct_chromas_wide - octave) < .5
        correct_chromas_wide = tf.logical_and(ref_voicing, correct_chromas_wide)
        correct_chromas_strict = tf.logical_and(est_voicing, correct_chromas_wide)
        correct_chromas_wide = count_nonzero_fn(correct_chromas_wide)
        correct_chromas_strict = count_nonzero_fn(correct_chromas_strict)
        self.update_melody_var_fn(rec_idx,  'correct_chromas', 'wide', correct_chromas_wide)
        self.update_melody_var_fn(rec_idx,  'correct_chromas', 'strict', correct_chromas_strict)

        assert all(self.var_dict['all_updated'].values())

        est_notes_with_voicing_info = tf.where(est_voicing, est_notes, -est_notes)

        return est_notes_with_voicing_info

    def update_states(self, rec_idx, logits):

        num_recs = self.num_recs
        assert rec_idx < num_recs

        track_ids = self.track_ids
        track_id = track_ids[rec_idx]
        ref_notes_dict = HardThresholdingInferencePerformance.gen_label_fn(track_id)

        ref_notes = ref_notes_dict['notes']

        assert logits.shape[0] == 721
        logits = logits.T
        logits = np.maximum(logits, 1e-11)
        logits = np.log10(logits) + 6.
        logits = np.require(logits, dtype=np.float32, requirements=['C'])
        viterbi_voicing, viterbi_bins = ViterbiInferencePerformance.viterbi_ins(logits)

        self.update_states_tf_fn(
            rec_idx=rec_idx,
            ref_notes=ref_notes,
            viterbi_est_voicing=viterbi_voicing,
            viterbi_est_bins=viterbi_bins
        )


if __name__ == '__main__':

    config = Config()
    model_name = 'test'
    imm_ins = IMM()
    track_ids = config.tvt_split_dict[model_name]
    num_recs = len(track_ids)
    hard_inf_ins = HardThresholdingInferencePerformance(config, model_name)
    viterbi_inf_ins = ViterbiInferencePerformance(config, model_name)

    for rec_idx in range(num_recs):
        track_id = track_ids[rec_idx]
        wav_file = os.path.join(os.environ['medleydb'], track_id, track_id + '_MIX.wav')
        logits = imm_ins.SIMM_fn(wav_file)
        hard_inf_ins.update_states(rec_idx, logits)
        viterbi_inf_ins.update_states(rec_idx, logits)
    logging.info('\nthresholding')
    hard_inf_ins.results()
    logging.info('\nviterbi')
    viterbi_inf_ins.results()



