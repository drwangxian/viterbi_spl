
DEBUG = False

import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import tensorflow as tf
tf.random.set_seed(2023)
import numpy as np
np.random.seed(2023)
import glob
import os
from tf_imm import IMM
from self_defined import load_np_array_from_file_fn
from self_defined import is_vocals_m2m3_fn
import medleydb as mdb
import librosa
import mir_eval
from self_defined import ArrayToTableTFFn


class Config:

    def __init__(self):

        self.debug = DEBUG

        self.methods = ('original', 'thresholding', 'viterbi')
        self.method = 'original'
        # self.model_names = ('training', 'validation', 'test')
        self.model_names = ('test',)

        self.tvt_split_dict = Config.get_dataset_split_fn()
        self.tvt_split_dict['test'] = Config.get_adc04_track_ids_fn()
        if self.debug:
            for k in self.tvt_split_dict:
                self.tvt_split_dict[k] = self.tvt_split_dict[k][:3]

        self.tb_dir = 'tb_original_inf_adc04'
        assert self.method in self.tb_dir
        if not self.debug:
            self.chk_if_tb_dir_exists_fn()

    def chk_if_tb_dir_exists_fn(self):
        assert self.tb_dir is not None
        is_tb_dir_exist = glob.glob('{}/'.format(self.tb_dir))
        if is_tb_dir_exist:
            assert False, 'directory {} already exists'.format(self.tb_dir)

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

    @staticmethod
    def get_adc04_track_ids_fn():

        wav_files = glob.glob(os.path.join(os.environ['adc04'], '*.wav'))
        track_ids = [os.path.basename(wav_file)[:-4] for wav_file in wav_files]
        assert len(track_ids) == 20

        return track_ids

    @staticmethod
    def get_mirex05_track_ids_fn():

        test_songlist = ["train01", "train02", "train03", "train04", "train05", "train06", "train07", "train08",
                         "train09"]

        assert len(test_songlist) == 9

        return test_songlist

    @staticmethod
    def get_mir1k_track_ids_fn():

        wav_files = os.path.join(os.environ['mir1k'], 'Wavfile', '*.wav')
        wav_files = glob.glob(wav_files)
        track_ids = [os.path.basename(wav_file)[:-4] for wav_file in wav_files]
        track_ids = set(track_ids)
        assert len(track_ids) == 1000
        track_ids = list(track_ids)
        track_ids = np.sort(track_ids)
        track_ids = list(track_ids)

        return track_ids

    @staticmethod
    def get_rwc_track_ids_fn():

        track_ids = []
        for rec_idx in range(100):
            track_ids.append(str(rec_idx))

        return track_ids


class Viterbi:

    THRESHOLD = 2.442347

    def __init__(self):

        self.num_freq_bins = 721
        self.h = 256
        self.sr = 44100
        self.single_side_peak_width = 20

        self.threshold = Viterbi.THRESHOLD
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
        logits = logits.T
        n_frames, n_freq_bins = logits.shape
        assert n_freq_bins == self.num_freq_bins
        frames_logits = logits
        frames_logits = np.require(frames_logits, np.float32, ['C'])

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


class AcousticModel:

    acoustic_model = IMM()
    logits_fn = acoustic_model.logits_fn
    melody_fn = acoustic_model.melody_fn
    viterbi_ins = Viterbi()


class TFDataset:

    def __init__(self, model):

        self.model = model

        self.wav_files = []
        self.np_dataset = self.gen_np_dataset_fn()
        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)

        num_frames = [len(rec_dict['notes']) for rec_dict in self.np_dataset]
        num_frames_vector = np.asarray(num_frames, dtype=np.int64)
        num_frames_vector.flags['WRITEABLE'] = False
        self.num_frames_vector = num_frames_vector

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
        TFDataset.validity_check_of_ref_freqs_fn(m2_freqs)

        vocal_freqs = np.where(is_vocals, m2_freqs, 0.)
        notes = TFDataset.hz_to_midi_fn(vocal_freqs)

        result = dict(notes=notes, original=dict(times=m2_melody[:, 0], freqs=vocal_freqs))

        return result

    def gen_np_dataset_fn(self):

        model = self.model

        logging.info('mdb - {} - generating labels'.format(model.name))

        track_ids = model.config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.debug('{}/{} - {}'.format(track_idx + 1, num_tracks, track_id))
            wav_file = os.path.join(os.environ['medleydb'], track_id, track_id + '_MIX.wav')
            self.wav_files.append(wav_file)
            labels = TFDataset.gen_label_fn(track_id)
            dataset.append(labels)

        return dataset


class TFDatasetForAdc04(TFDataset):

    def __init__(self, model):
        super(TFDatasetForAdc04, self).__init__(model)

    @staticmethod
    def gen_label_fn(track_id):
        # the reference melody uses a hop size of 256 samples
        melody2_suffix = 'REF.txt'

        annot_path = os.path.join(os.environ['adc04'], track_id + melody2_suffix)
        times_labels = np.genfromtxt(annot_path, delimiter=None)
        assert times_labels.ndim == 2 and times_labels.shape[1] == 2
        assert np.all(np.logical_not(np.isnan(times_labels)))
        num_frames = len(times_labels)
        t = times_labels[-1, 0]
        t = int(round(t / (256. / 44100.)))
        assert t + 1 == num_frames
        assert times_labels[0, 0] == 0.

        freqs = times_labels[:, 1]
        TFDataset.validity_check_of_ref_freqs_fn(freqs)

        notes = TFDataset.hz_to_midi_fn(freqs)

        return dict(notes=notes, original=dict(times=times_labels[:, 0], freqs=times_labels[:, 1]))

    def gen_np_dataset_fn(self):

        model = self.model

        logging.info('adc04 - {} - generating labels'.format(model.name))

        track_ids = model.config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.debug('{}/{} - {}'.format(track_idx + 1, num_tracks, track_id))
            wav_file = os.path.join(os.environ['adc04'], track_id + '.wav')
            self.wav_files.append(wav_file)
            labels = TFDatasetForAdc04.gen_label_fn(track_id)
            dataset.append(labels)

        return dataset


class MetricsBase:

    def __init__(self, model):

        self.model = model
        self.num_recs = len(model.config.tvt_split_dict[model.name])
        self.var_dict = self.define_tf_variables_fn()
        self.mir_eval_oas = []
        self.tf_oas = None

    def define_tf_variables_fn(self):

        num_recs = self.num_recs

        with tf.name_scope(self.model.name):
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
    def octave(distance):

        distance = tf.floor(distance / 12. + .5) * 12.

        return distance

    @staticmethod
    def count_nonzero_fn(inputs):

        outputs = inputs
        outputs = tf.math.count_nonzero(outputs, dtype=tf.int32)
        outputs = tf.cast(outputs, tf.int64)

        return outputs

    @staticmethod
    def est_notes_to_hz_for_mir_eval_fn(est_notes):

        positives = np.where(est_notes > 1)
        freqs = np.zeros_like(est_notes)
        freqs[positives] = librosa.midi_to_hz(est_notes[positives])

        return freqs

    def mir_eval_oa_fn(self, rec_idx, est_notes):

        model = self.model

        rec_dict = model.dataset.np_dataset[rec_idx]
        ref_times = rec_dict['original']['times']
        ref_freqs = rec_dict['original']['freqs']
        assert len(ref_times) == len(ref_freqs)

        est_freqs = MetricsBase.est_notes_to_hz_for_mir_eval_fn(est_notes)

        if np.all(est_freqs == 0.):
            logging.warning('{} - all frames unvoiced'.format(rec_idx))

        num_frames = len(est_freqs)
        h = 256. / 44100
        est_times = np.arange(num_frames) * h

        oa = mir_eval.melody.evaluate(
            ref_time=ref_times,
            ref_freq=ref_freqs,
            est_time=est_times,
            est_freq=est_freqs
        )['Overall Accuracy']

        return oa

    @staticmethod
    def to_f8_divide_and_to_f4_fn(numerator, denominator):

        numerator = tf.cast(numerator, tf.float64)
        denominator = tf.cast(denominator, tf.float64)
        numerator = numerator / tf.maximum(denominator, 1e-7)
        numerator = tf.cast(numerator, tf.float32)

        return numerator

    def results(self):

        model = self.model
        num_recs = self.num_recs
        melody_dict = self.var_dict['melody']

        f8f4div = MetricsBase.to_f8_divide_and_to_f4_fn

        num_frames_vector = tf.convert_to_tensor(model.dataset.num_frames_vector, tf.int64)
        _num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        tf.debugging.assert_equal(_num_frames_vector, num_frames_vector)

        correct_frames = melody_dict['correct_pitches']['strict'] + melody_dict['voicing']['correct_unvoiced']
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

        self.tf_oas = m_oa.numpy()
        assert len(self.mir_eval_oas) == num_recs

        results = dict(
            vrr=m_vrr,
            vfa=m_vfa,
            va=m_va,
            rpa_strict=m_rpa_strict,
            rpa_wide=m_rpa_wide,
            rca_strict=m_rca_strict,
            rca_wide=m_rca_wide,
            oa=m_oa
        )

        return results


class MetricsOriginal(MetricsBase):

    def __init__(self, model):

        super(MetricsOriginal, self).__init__(model)

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.int32, name='rec_idx'),
        tf.TensorSpec([None], name='ref_notes'),
        tf.TensorSpec([None], tf.bool, name='est_voicing'),
        tf.TensorSpec([None], tf.int32, name='est_bins')
    ], autograph=False)
    def update_states_tf_fn(self, rec_idx, ref_notes, est_voicing, est_bins):

        count_nonzero_fn = MetricsOriginal.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        rec_idx.set_shape([])

        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        ref_notes.set_shape([None])

        viterbi_est_voicing = tf.convert_to_tensor(est_voicing, tf.bool)
        viterbi_est_voicing.set_shape([None])

        viterbi_est_bins = tf.convert_to_tensor(est_bins, tf.int32)
        viterbi_est_bins.set_shape([None])

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_notes = MetricsOriginal.est_notes_fn(viterbi_est_bins)
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
        octave = MetricsOriginal.octave(correct_chromas_wide)
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

    def update_states_fn(self, rec_idx, f0s):

        num_recs = self.num_recs
        model = self.model

        assert rec_idx < num_recs
        ref_dict = model.dataset.np_dataset[rec_idx]
        ref_notes = ref_dict['notes']

        n_frames = len(ref_notes)
        _n_frames = len(f0s)
        t = n_frames - _n_frames
        if np.abs(t) > 1:
            logging.warning('large difference - {} frames'.format(t))

        oa = self.mir_eval_oa_fn(rec_idx=rec_idx, f0s=f0s)
        self.mir_eval_oas.append(oa)

    def mir_eval_oa_fn(self, rec_idx, f0s):

        model = self.model

        rec_dict = model.dataset.np_dataset[rec_idx]
        ref_times = rec_dict['original']['times']
        ref_freqs = rec_dict['original']['freqs']
        assert len(ref_times) == len(ref_freqs)

        est_freqs = f0s

        if np.all(est_freqs == 0.):
            logging.warning('{} - all frames unvoiced'.format(rec_idx))

        num_frames = len(est_freqs)
        h = 256. / 44100
        est_times = np.arange(num_frames) * h

        oa = mir_eval.melody.evaluate(
            ref_time=ref_times,
            ref_freq=ref_freqs,
            est_time=est_times,
            est_freq=est_freqs
        )['Overall Accuracy']

        return oa


class MetricsThresholding(MetricsBase):

    def __init__(self, model):

        super(MetricsThresholding, self).__init__(model)

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.int32, name='rec_idx'),
        tf.TensorSpec([None], name='ref_notes'),
        tf.TensorSpec([721, None], name='logits')
    ], autograph=False)
    def update_states_tf_fn(self, rec_idx, ref_notes, logits):

        count_nonzero_fn = MetricsOriginal.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        rec_idx.set_shape([])
        logits = tf.convert_to_tensor(logits, tf.float32)
        logits.set_shape([721, None])

        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        ref_notes.set_shape([None])

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        logits = tf.transpose(logits)
        logits_argmax = tf.argmax(logits, axis=1, output_type=tf.int32)
        est_notes = MetricsBase.est_notes_fn(logits_argmax)
        logits_peaks = tf.gather(logits, axis=1, batch_dims=1, indices=logits_argmax)
        est_voicing = logits_peaks > Viterbi.THRESHOLD
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
        octave = MetricsOriginal.octave(correct_chromas_wide)
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

    def update_states_fn(self, rec_idx, logits):

        num_recs = self.num_recs
        model = self.model

        assert rec_idx < num_recs
        assert logits.shape[0] == 721

        ref_dict = model.dataset.np_dataset[rec_idx]
        ref_notes = ref_dict['notes']

        n_frames = len(ref_notes)
        _n_frames = logits.shape[1]
        t = n_frames - _n_frames
        if np.abs(t) > 1:
            logging.warning('large difference - {} frames'.format(t))
        if t > 0:
            logits = np.pad(logits, [[0, 0], [0, t]], mode='constant', constant_values=-5.)
        elif t < 0:
            ref_notes = np.pad(ref_notes, [[0, -t]])
        assert logits.shape[1] == len(ref_notes)

        est_notes_with_voicing_info = self.update_states_tf_fn(
            rec_idx=rec_idx,
            ref_notes=ref_notes,
            logits=logits
        )
        est_notes_with_voicing_info = est_notes_with_voicing_info.numpy()

        oa = self.mir_eval_oa_fn(
            rec_idx=rec_idx,
            est_notes=est_notes_with_voicing_info
        )
        self.mir_eval_oas.append(oa)


class MetricsViterbi(MetricsOriginal):

    def __init__(self, model):

        super(MetricsViterbi, self).__init__(model)

    def viterbi_then_update_states_fn(self, rec_idx, logits):

        assert logits.shape[0] == 721

        voicing, bins = AcousticModel.viterbi_ins(logits)
        self.update_states_fn(rec_idx=rec_idx, voicing=voicing, bins=bins)


class TBSummary:

    def __init__(self, model):

        assert hasattr(model, 'metrics')

        self.model = model

        self.tb_path = os.path.join(model.config.tb_dir, model.name)
        self.tb_summary_writer = tf.summary.create_file_writer(self.tb_path)
        self.rec_names = model.dataset.rec_names
        self.num_recs = len(self.rec_names)

        self.header = ['vrr', 'vfa', 'va', 'rpa_strict', 'rpa_wide', 'rca_strict', 'rca_wide', 'oa']
        self.num_columns = len(self.header)

        self.table_ins = self.create_tf_table_writer_ins_fn()

    def create_tf_table_writer_ins_fn(self):

        header = self.header
        description = 'metrics'
        tb_summary_writer = self.tb_summary_writer

        names = list(self.rec_names) + ['average']
        table_ins = ArrayToTableTFFn(
            writer=tb_summary_writer,
            header=header,
            scope=description,
            title=description,
            names=names
        )

        return table_ins

    def prepare_table_data_fn(self, result_dict):

        header = self.header
        data = [result_dict[name] for name in header]
        data = tf.stack(data, axis=-1)
        tf.ensure_shape(data, [self.num_recs, self.num_columns])
        ave = tf.reduce_mean(data, axis=0, keepdims=True)
        data = tf.concat([data, ave], axis=0)

        return data

    def write_tb_summary_fn(self, step_int):

        model = self.model

        assert isinstance(step_int, int)

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                result_dict = model.metrics.results()
                data = self.prepare_table_data_fn(result_dict)
                self.table_ins.write(data, step_int)


class Model:

    def __init__(self, config, name):

        assert name in config.model_names
        self.name = name
        self.config = config

        if name == 'test':
            self.dataset = TFDatasetForAdc04(self)
        else:
            self.dataset = TFDataset(self)

        method = config.method
        if method == 'original':
            self.metrics = MetricsOriginal(self)
        elif method == 'thresholding':
            self.metrics = MetricsThresholding(self)
        elif method == 'viterbi':
            self.metrics = MetricsViterbi(self)
        else:
            assert False

        self.tb_summary_ins = TBSummary(self)


def main():

    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    for name in MODEL_DICT['config'].model_names:
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    aug_info = []
    aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug))
    aug_info.append('method - {}'.format(MODEL_DICT['config'].method))
    aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
    aug_info = '\n\n'.join(aug_info)
    logging.info('\n' + aug_info)
    t = MODEL_DICT['config'].model_names[-1]
    with MODEL_DICT[t].tb_summary_ins.tb_summary_writer.as_default():
        tf.summary.text('auxiliary_information', aug_info, step=0)

    def inference_fn(model_name, global_step=None):

        assert isinstance(global_step, int)

        config = MODEL_DICT['config']
        model = MODEL_DICT[model_name]
        metrics = model.metrics
        wav_files = model.dataset.wav_files
        method = config.method

        if method == 'original':
            for rec_idx, wav_file in enumerate(wav_files):
                melody_file = os.path.basename(wav_file)[:-4] + '.npz'
                melody_file = os.path.join('../imm_original', melody_file)
                f0s = np.load(melody_file)['f0s']
                assert not np.any(np.isnan(f0s))
                assert not np.any(np.isneginf(f0s))
                metrics.update_states_fn(rec_idx=rec_idx, f0s=f0s)
        elif method == 'thresholding':
            for rec_idx, wav_file in enumerate(wav_files):
                logits = AcousticModel.logits_fn(wav_file)
                metrics.update_states_fn(rec_idx=rec_idx, logits=logits)
        elif method == 'viterbi':
            for rec_idx, wav_file in enumerate(wav_files):
                logits = AcousticModel.logits_fn(wav_file)
                metrics.viterbi_then_update_states_fn(rec_idx=rec_idx, logits=logits)

        oas = metrics.mir_eval_oas
        oas = np.asarray(oas)
        ave_oa = np.mean(oas)
        for idx, oa in enumerate(oas):
            print('{} - {}'.format(idx, oa))
        print('ave - {}'.format(ave_oa))

    for model_name in MODEL_DICT['config'].model_names:
        inference_fn(model_name=model_name, global_step=0)
        model = MODEL_DICT[model_name]
        model.tb_summary_ins.tb_summary_writer.close()


if __name__ == '__main__':
    main()






