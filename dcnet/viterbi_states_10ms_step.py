"""
jdc used a step of 10 ms.
In this code, we calculate parameters for the viterbi algorithm applied to jdc.







"""

DEBUG = False
GPU_ID = 1

import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import tensorflow as tf
from argparse import Namespace
import glob
import os
import numpy as np
import librosa
from self_defined import ArrayToTableTFFn
import acoustic_model_shaun as acoustic_model_module
import nsgt as nsgt_module
assert not nsgt_module.USE_DOUBLE
import soundfile
import mir_eval
import medleydb as mdb
from self_defined import is_vocals_m2m3_fn as is_vocals_fn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from self_defined import save_np_array_to_file_fn, load_np_array_from_file_fn


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

        self.debug_mode = DEBUG


        self.validation_split_dict = Config.get_dataset_split_fn()['validation']
        if self.debug_mode:
            self.validation_split_dict = self.validation_split_dict[:2]






class TFDataset:

    hop_size = 441

    medleydb_dir = os.environ['medleydb']
    melody2_dir = os.environ['melody2_dir']
    mix_suffix = '_MIX.wav'
    melody2_suffix = '_MELODY2.csv'
    min_melody_freq = librosa.midi_to_hz(23.6)

    note_range = np.arange(721) / 16. + 38.
    assert np.isclose(note_range[0], 38)
    assert np.isclose(note_range[-1], 83)
    note_range.flags['WRITEABLE'] = False
    note_min = 38
    freq_min = librosa.midi_to_hz(note_min)

    def __init__(self, model):

        self.model = model

    @staticmethod
    def spec_transform_fn(spec):

        top_db = 120.
        spec = librosa.amplitude_to_db(spec, ref=np.max, amin=1e-7, top_db=top_db)
        spec = spec / top_db + 1.
        spec = spec.astype(np.float32)

        return spec

    @staticmethod
    def gen_spec_fn(track_id):

        nsgt_instances = TFDataset.nsgt_instances
        nsgt_Lses = TFDataset.nsgt_Lses
        hop_size = TFDataset.nsgt_hop_size
        num_freq_bins = TFDataset.nsgt_freq_bins
        medleydb_dir = TFDataset.medleydb_dir
        mix_suffix = TFDataset.mix_suffix

        mix_wav_file = os.path.join(medleydb_dir, track_id, track_id + mix_suffix)
        num_samples = soundfile.info(mix_wav_file).frames
        t = np.searchsorted(nsgt_Lses, num_samples)
        assert 1 <= t <= len(nsgt_instances)
        nsgt_fn = nsgt_instances[t - 1].nsgt_of_wav_file_fn
        nsgt = nsgt_fn(mix_wav_file)
        num_frames = (num_samples + hop_size - 1) // hop_size
        assert nsgt.shape == (num_frames, num_freq_bins)
        nsgt = nsgt[::4, 1:501]
        nsgt = TFDataset.spec_transform_fn(nsgt)
        nsgt = np.require(nsgt, np.float32, requirements=['O', 'C'])

        return nsgt

    @staticmethod
    def hz_to_midi_fn(freqs):

        notes = np.zeros_like(freqs)
        positives = np.nonzero(freqs)
        notes[positives] = librosa.hz_to_midi(freqs[positives])

        return notes

    @staticmethod
    def midi_to_hz_fn(notes):

        assert np.all(notes >= 0)

        freqs = np.zeros_like(notes)
        positives = np.where(notes > 0)
        freqs[positives] = librosa.midi_to_hz(notes[positives])

        return freqs

    @staticmethod
    def gen_label_fn(track_id):

        track = mdb.MultiTrack(track_id)
        assert not track.is_instrumental
        assert track.has_melody

        m2_melody = track.melody2_annotation
        m2_melody = np.asarray(m2_melody)
        num_frames = len(m2_melody)

        is_vocals = is_vocals_fn(track_id)
        assert len(is_vocals) == num_frames

        m2_times = m2_melody[:, 0]
        m2_freqs = m2_melody[:, 1]
        TFDataset.validity_check_of_ref_freqs_fn(m2_freqs)

        vocal_freqs = np.where(is_vocals, m2_freqs, 0.)

        num_frames_10ms = ((num_frames - 1) * 256 + 440) // 441 + 1
        times_new = np.arange(num_frames_10ms) * 0.010
        assert times_new[-1] >= m2_times[-1]
        vocal_freqs_10ms, _ = mir_eval.melody.resample_melody_series(
            times=m2_times,
            frequencies=vocal_freqs,
            voicing=vocal_freqs > .1,
            times_new=times_new
        )
        TFDataset.validity_check_of_ref_freqs_fn(vocal_freqs_10ms)
        notes = TFDataset.hz_to_midi_fn(vocal_freqs_10ms)

        result = dict(notes=notes, original=dict(times=m2_times, freqs=vocal_freqs))

        return result

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - MedleyDB - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.debug('{}/{}'.format(track_idx + 1, num_tracks))

            spec = TFDataset.gen_spec_fn(track_id)
            notes_original_dict = TFDataset.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - len(spec)
            assert 0 <= diff <= 1
            if diff == 1:
                spec = np.pad(spec, [[0, 1], [0, 0]])

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        return dataset

    def note_out_of_range_chk_fn(self, np_dataset):

        model = self.model
        note_range = TFDataset.note_range

        logging.info('{} - note range checking ... '.format(model.name))
        note_min = min(note for rec_dict in np_dataset for note in rec_dict['notes'] if note > 0)
        note_max = max(note for rec_dict in np_dataset for note in rec_dict['notes'] if note > 0)

        lower_note = note_range[0]
        upper_note = note_range[-1]
        logging.info('note range - ({}, {})'.format(lower_note, upper_note))
        if note_min < lower_note or note_min > upper_note:
            logging.warning('note min - {} - out of range'.format(note_min))
        if note_max < lower_note or note_max > upper_note:
            logging.warning('note max - {} - out of range'.format(note_max))

    @staticmethod
    def gen_split_list_fn(num_frames, snippet_len):

        split_frames = range(0, num_frames + 1, snippet_len)
        split_frames = list(split_frames)
        if split_frames[-1] != num_frames:
            split_frames.append(num_frames)
        start_end_frame_pairs = zip(split_frames[:-1], split_frames[1:])
        start_end_frame_pairs = [list(it) for it in start_end_frame_pairs]

        return start_end_frame_pairs

    @staticmethod
    def validity_check_of_ref_freqs_fn(freqs):

        min_melody_freq = TFDataset.min_melody_freq

        all_zeros = freqs == 0.
        all_positives = freqs > min_melody_freq
        all_valid = np.logical_or(all_zeros, all_positives)
        assert np.all(all_valid)


class TFDatasetForTrainingModeTrainingSplit(TFDataset):

    def __init__(self, model):

        super(TFDatasetForTrainingModeTrainingSplit, self).__init__(model)

        is_inferencing = model.config.train_or_inference.inference is not None
        assert not is_inferencing
        assert model.is_training
        assert 'train' in model.name

        self.np_dataset = self.gen_np_dataset_fn()
        self.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_rec_start_end_idx_fn(self.np_dataset)
        self.tf_dataset = self.gen_tf_dataset_fn()
        self.iterator = iter(self.tf_dataset)
        self.set_num_batches_per_epoch_fn()

    def gen_rec_start_end_idx_fn(self, np_dataset):

        snippet_len = self.model.config.snippet_len
        rec_start_and_end_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=len(rec_dict['spectrogram']),
                snippet_len=snippet_len
            )
            tmp = [[rec_idx] + se for se in split_list]
            rec_start_and_end_list.extend(tmp)

        return rec_start_and_end_list

    def map_idx_to_data_fn(self, idx):

        def py_fn(idx):

            idx = idx.numpy().item()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = self.np_dataset[rec_idx]
            spec = rec_dict['spectrogram'][start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            return spec, notes

        spec, notes = tf.py_function(py_fn, inp=[idx], Tout=['float32', 'float32'])
        spec.set_shape([None, 500])
        notes.set_shape([None])

        return dict(spectrogram=spec, notes=notes)

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.shuffle(num_snippets, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(10)

        return dataset

    def set_num_batches_per_epoch_fn(self):

        model = self.model
        if model.config.batches_per_epoch is None:
            batches_per_epoch = len(self.rec_start_end_idx_list)
            model.config.batches_per_epoch = batches_per_epoch
            logging.info('batches per epoch set to {}'.format(batches_per_epoch))


class TFDatasetForInferenceMode(TFDataset):

    def __init__(self, model):

        super(TFDatasetForInferenceMode, self).__init__(model)

        is_inferencing = model.config.train_or_inference.inference is not None

        if not is_inferencing:
            assert not model.is_training
            assert 'validation' in model.name

        self.np_dataset = self.gen_np_dataset_fn()

        # if is_inferencing and model.name == 'validation':
        #     self.validation_stats_fn()
        #     assert False


        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)
        num_frames = [len(rec_dict['spectrogram']) for rec_dict in self.np_dataset]
        self.num_frames_vector = np.asarray(num_frames, dtype=np.int64)
        self.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_split_list_fn(self.np_dataset)
        self.tf_dataset = self.gen_tf_dataset_fn()

    def gen_split_list_fn(self, np_dataset):

        model = self.model
        snippet_len = model.config.snippet_len

        rec_start_end_idx_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=len(rec_dict['spectrogram']),
                snippet_len=snippet_len
            )
            rec_dict['split_list'] = split_list
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list.extend(t)

        return rec_start_end_idx_list

    def map_idx_to_data_fn(self, idx):

        def py_fn(idx):

            idx = idx.numpy().item()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = self.np_dataset[rec_idx]
            snippet_len = self.model.config.snippet_len
            assert start_frame % snippet_len == 0
            snippet_idx = start_frame // snippet_len
            spec = rec_dict['spectrogram'][start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            return rec_idx, snippet_idx, spec, notes

        rec_idx, snippet_idx, spec, notes = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.float32, tf.float32]
        )
        rec_idx.set_shape([])
        snippet_idx.set_shape([])
        spec.set_shape([None, 500])
        notes.set_shape([None])

        return dict(
            rec_idx=rec_idx,
            snippet_idx=snippet_idx,
            spectrogram=spec,
            notes=notes
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(10)

        return dataset

    def ref_notes_quantization_fn(self, ref_notes):

        min_note, max_note = TFDataset.note_range[[0, -1]]

        ref_notes = np.minimum(ref_notes, max_note)
        _min_note = np.min(ref_notes[ref_notes > 1])
        assert _min_note >= min_note

        ref_notes = (ref_notes - min_note) * 5
        ref_notes = np.round(ref_notes)
        ref_notes = ref_notes.astype(np.int32)
        ref_notes[ref_notes < 0] = 320

        return ref_notes

    def validation_stats_fn(self):

        model = self.model
        assert model.name == 'validation'
        assert model.config.train_or_inference.inference is not None

        dataset = self.np_dataset

        n_bins = 320
        p_steady = np.zeros([n_bins + 1], np.int64)
        transition = np.zeros([n_bins + 1, n_bins + 1], np.int64)
        switch = np.zeros([2, 2], np.int64)

        n_recs = len(dataset)
        for rec_idx in range(n_recs):
            ref_notes = dataset[rec_idx]['notes']
            ref_notes = self.ref_notes_quantization_fn(ref_notes)
            unvoiced = ref_notes == n_bins
            unvoiced = unvoiced.astype(np.int32)

            for note, _note in zip(ref_notes[:-1], ref_notes[1:]):
                p_steady[note] += 1
                transition[note, _note] += 1
            p_steady[_note] += 1

            for uv, uv_ in zip(unvoiced[:-1], unvoiced[1:]):
                switch[uv, uv_] += 1

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


class TFDatasetForAdc04(TFDataset):

    def __init__(self, model):

        super(TFDatasetForAdc04, self).__init__(model)

        is_inferencing = model.config.train_or_inference.inference is not None

        assert is_inferencing

        self.np_dataset = self.gen_np_dataset_fn()

        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)
        num_frames = [len(rec_dict['spectrogram']) for rec_dict in self.np_dataset]
        self.num_frames_vector = np.asarray(num_frames, dtype=np.int64)
        self.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_split_list_fn(self.np_dataset)
        self.tf_dataset = self.gen_tf_dataset_fn()

    @staticmethod
    def gen_spec_fn(track_id):

        nsgt_instances = TFDataset.nsgt_instances
        nsgt_Lses = TFDataset.nsgt_Lses
        hop_size = TFDataset.nsgt_hop_size
        num_freq_bins = TFDataset.nsgt_freq_bins

        mix_wav_file = os.path.join(os.environ['adc04'], track_id + '.wav')
        info = soundfile.info(mix_wav_file)
        assert info.samplerate == 44100
        assert info.subtype == 'PCM_16'
        num_samples = info.frames
        t = np.searchsorted(nsgt_Lses, num_samples)
        assert 1 <= t <= len(nsgt_instances)
        nsgt_fn = nsgt_instances[t - 1].nsgt_of_wav_file_fn
        nsgt = nsgt_fn(mix_wav_file)
        num_frames = (num_samples + hop_size - 1) // hop_size
        assert nsgt.shape == (num_frames, num_freq_bins)
        nsgt = nsgt[::4, 1:501]
        nsgt = TFDataset.spec_transform_fn(nsgt)
        nsgt = np.require(nsgt, np.float32, requirements=['O', 'C'])

        return nsgt

    @staticmethod
    def gen_label_fn(track_id):
        # the reference melody uses a hop size of 256 samples
        melody2_suffix = 'REF.txt'

        annot_path = os.path.join(os.environ['adc04'], track_id + melody2_suffix)
        times_labels = np.genfromtxt(annot_path, delimiter=None)
        assert np.all(np.logical_not(np.isnan(times_labels)))
        assert times_labels.ndim == 2 and times_labels.shape[1] == 2
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

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - adc04 - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.debug('{}/{}'.format(track_idx + 1, num_tracks))

            spec = TFDatasetForAdc04.gen_spec_fn(track_id)
            notes_original_dict = TFDatasetForAdc04.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - len(spec)
            assert 0 <= diff <= 1
            if diff == 1:
                spec = np.pad(spec, [[0, 1], [0, 0]])

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        return dataset

    def gen_split_list_fn(self, np_dataset):

        model = self.model
        snippet_len = model.config.snippet_len

        rec_start_end_idx_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=len(rec_dict['spectrogram']),
                snippet_len=snippet_len
            )
            rec_dict['split_list'] = split_list
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list.extend(t)

        return rec_start_end_idx_list

    def map_idx_to_data_fn(self, idx):

        def py_fn(idx):

            idx = idx.numpy().item()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = self.np_dataset[rec_idx]
            snippet_len = self.model.config.snippet_len
            assert start_frame % snippet_len == 0
            snippet_idx = start_frame // snippet_len
            spec = rec_dict['spectrogram'][start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            return rec_idx, snippet_idx, spec, notes

        rec_idx, snippet_idx, spec, notes = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.float32, tf.float32]
        )
        rec_idx.set_shape([])
        snippet_idx.set_shape([])
        spec.set_shape([None, 500])
        notes.set_shape([None])

        return dict(
            rec_idx=rec_idx,
            snippet_idx=snippet_idx,
            spectrogram=spec,
            notes=notes
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(10)

        return dataset


class TFDatasetForMirex05(TFDataset):

    def __init__(self, model):

        super(TFDatasetForMirex05, self).__init__(model)

        is_inferencing = model.config.train_or_inference.inference is not None
        assert is_inferencing

        self.np_dataset = self.gen_np_dataset_fn()
        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)
        num_frames = [len(rec_dict['spectrogram']) for rec_dict in self.np_dataset]
        self.num_frames_vector = np.asarray(num_frames, dtype=np.int64)
        self.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_split_list_fn(self.np_dataset)
        self.tf_dataset = self.gen_tf_dataset_fn()

    @staticmethod
    def gen_spec_fn(track_id):

        nsgt_instances = TFDataset.nsgt_instances
        nsgt_Lses = TFDataset.nsgt_Lses
        hop_size = TFDataset.nsgt_hop_size
        num_freq_bins = TFDataset.nsgt_freq_bins

        mix_wav_file = os.path.join(os.environ['mirex05'], track_id + '.wav')
        info = soundfile.info(mix_wav_file)
        assert info.samplerate == 44100
        assert info.subtype == 'PCM_16'
        num_samples = info.frames
        t = np.searchsorted(nsgt_Lses, num_samples)
        assert 1 <= t <= len(nsgt_instances)
        nsgt_fn = nsgt_instances[t - 1].nsgt_of_wav_file_fn
        nsgt = nsgt_fn(mix_wav_file)
        num_frames = (num_samples + hop_size - 1) // hop_size
        assert nsgt.shape == (num_frames, num_freq_bins)
        nsgt = nsgt[::4, 1:501]
        nsgt = TFDataset.spec_transform_fn(nsgt)
        nsgt = np.require(nsgt, np.float32, requirements=['O', 'C'])

        return nsgt

    @staticmethod
    def gen_label_fn(track_id):
        # reference melody uses a hop size of 441 samples, or 10 ms

        if track_id == 'train13MIDI':
            m2_file = os.path.join(os.environ['mirex05'], 'train13REF.txt')
        else:
            m2_file = os.path.join(os.environ['mirex05'], track_id + 'REF.txt')
        times_labels = np.genfromtxt(m2_file, delimiter=None)
        assert np.all(np.logical_not(np.isnan(times_labels)))
        assert times_labels.ndim == 2 and times_labels.shape[1] == 2
        num_frames = len(times_labels)
        t = times_labels[-1, 0]
        t = int(round(t / .01))
        assert t + 1 == num_frames
        assert times_labels[0, 0] == 0.
        freqs_441 = times_labels[:, 1]
        TFDataset.validity_check_of_ref_freqs_fn(freqs_441)

        num_frames_256 = ((num_frames - 1) * 441 + 255) // 256 + 1
        times_256 = np.arange(num_frames_256) * (256. / 44100.)
        times_441 = np.arange(num_frames) * 0.01
        assert times_256[-1] >= times_441[-1]
        voicing_441 = freqs_441 > .1
        freqs_256, _ = mir_eval.melody.resample_melody_series(
            times=times_441,
            frequencies=freqs_441,
            voicing=voicing_441,
            times_new=times_256
        )
        TFDataset.validity_check_of_ref_freqs_fn(freqs_256)
        notes = TFDataset.hz_to_midi_fn(freqs_256)

        result = dict(notes=notes, original=dict(times=times_labels[:, 0], freqs=freqs_441))

        return result

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - Mirex05 training - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.info('{}/{} - {}'.format(track_idx + 1, num_tracks, track_id))

            spec = TFDatasetForMirex05.gen_spec_fn(track_id)
            notes_original_dict = TFDatasetForMirex05.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - len(spec)
            logging.info('  diff - {}'.format(diff))
            if diff > 0:
                spec = np.pad(spec, [[0, diff], [0, 0]])
            elif diff < 0:
                notes = np.pad(notes, [[0, -diff]])

            assert len(notes) == len(spec)

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        return dataset

    def gen_split_list_fn(self, np_dataset):

        model = self.model
        snippet_len = model.config.snippet_len

        rec_start_end_idx_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=len(rec_dict['spectrogram']),
                snippet_len=snippet_len
            )
            rec_dict['split_list'] = split_list
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list.extend(t)

        return rec_start_end_idx_list

    def map_idx_to_data_fn(self, idx):

        def py_fn(idx):

            idx = idx.numpy().item()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = self.np_dataset[rec_idx]
            snippet_len = self.model.config.snippet_len
            assert start_frame % snippet_len == 0
            snippet_idx = start_frame // snippet_len
            spec = rec_dict['spectrogram'][start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            return rec_idx, snippet_idx, spec, notes

        rec_idx, snippet_idx, spec, notes = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.float32, tf.float32]
        )
        rec_idx.set_shape([])
        snippet_idx.set_shape([])
        spec.set_shape([None, 500])
        notes.set_shape([None])

        return dict(
            rec_idx=rec_idx,
            snippet_idx=snippet_idx,
            spectrogram=spec,
            notes=notes
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(10)

        return dataset


class TFDatasetForMir1k(TFDataset):

    def __init__(self, model):

        super(TFDatasetForMir1k, self).__init__(model)

        is_inferencing = model.config.train_or_inference.inference is not None
        assert is_inferencing

        self.np_dataset = self.gen_np_dataset_fn()
        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)
        num_frames = [len(rec_dict['spectrogram']) for rec_dict in self.np_dataset]
        self.num_frames_vector = np.asarray(num_frames, dtype=np.int64)
        self.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_split_list_fn(self.np_dataset)
        self.tf_dataset = self.gen_tf_dataset_fn()

    @staticmethod
    def gen_spec_fn(track_id):

        nsgt_instances = TFDataset.nsgt_instances
        nsgt_Lses = TFDataset.nsgt_Lses
        hop_size = TFDataset.nsgt_hop_size
        num_freq_bins = TFDataset.nsgt_freq_bins
        sr = 44100

        mix_wav_file = os.path.join(os.environ['mir1k'], 'Wavfile', track_id + '.wav')
        samples, _sr = librosa.load(mix_wav_file, sr=sr, dtype=np.float64)
        assert _sr == sr
        assert samples.ndim == 1
        assert samples.max() < 1
        assert samples.min() >= -1
        samples = samples.astype(np.float32)
        num_samples = len(samples)
        t = np.searchsorted(nsgt_Lses, num_samples)
        assert 1 <= t <= len(nsgt_instances)
        nsgt_fn = nsgt_instances[t - 1].nsgt_of_wav_file_fn
        nsgt = nsgt_fn(samples)
        num_frames = (num_samples + hop_size - 1) // hop_size
        assert nsgt.shape == (num_frames, num_freq_bins)
        nsgt = nsgt[::4, 1:501]
        nsgt = TFDataset.spec_transform_fn(nsgt)
        nsgt = np.require(nsgt, np.float32, requirements=['O', 'C'])

        return nsgt

    @staticmethod
    def gen_label_fn(track_id):
        # reference melody uses a hop size 20 ms. the staring time is 20 ms instead of 0 ms.

        wav_file = os.path.join(os.environ['mir1k'], 'Wavfile', track_id + '.wav')
        wav_info = soundfile.info(wav_file)
        assert wav_info.samplerate == 16000
        num_samples = wav_info.frames

        pitch_file = os.path.join(os.environ['mir1k'], 'PitchLabel', track_id + '.pv')
        pitches = np.genfromtxt(pitch_file)
        assert np.all(np.logical_not(np.isnan(pitches)))
        assert pitches.ndim == 1
        num_frames = len(pitches)
        w = 640
        assert num_samples >= w
        _num_frames = (num_samples - w) // 320 + 1
        assert num_frames == _num_frames
        t1 = pitches > 23
        t2 = pitches == 0
        assert np.all(np.logical_or(t1, t2))

        num_frames = num_frames + 1
        times_20ms = np.arange(num_frames) * .02
        pitches = np.pad(pitches, [[1, 0]])
        assert len(pitches) == len(times_20ms) == num_frames

        num_frames_256 = ((num_frames - 1) * 441 + 127) // 128 + 1
        times_256 = np.arange(num_frames_256) * (256. / 44100)
        assert times_256[-1] >= times_20ms[-1]
        pitches_256, _ = mir_eval.melody.resample_melody_series(
            times=times_20ms,
            frequencies=pitches,
            voicing=pitches > .1,
            times_new=times_256
        )
        t1 = pitches_256 == 0
        t2 = pitches_256 > 23
        assert np.all(np.logical_or(t1, t2))

        freqs = TFDataset.midi_to_hz_fn(pitches)

        result = dict(notes=pitches_256, original=dict(times=times_20ms, freqs=freqs))

        return result

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - mir1k - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        diffs = []
        for track_idx, track_id in enumerate(track_ids):
            logging.info('{}/{} - {}'.format(track_idx + 1, num_tracks, track_id))

            spec = TFDatasetForMir1k.gen_spec_fn(track_id)
            notes_original_dict = TFDatasetForMir1k.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - len(spec)
            if diff not in diffs:
                diffs.append(diff)
            if diff > 0:
                spec = np.pad(spec, [[0, diff], [0, 0]])
            elif diff < 0:
                notes = np.pad(notes, [[0, -diff]])

            assert len(notes) == len(spec)

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        diffs = np.asarray(diffs)
        max_diff = diffs.max()
        min_diff = diffs.min()
        logging.info('diff - max - {} - min - {}'.format(max_diff, min_diff))

        return dataset

    def gen_split_list_fn(self, np_dataset):

        model = self.model
        snippet_len = model.config.snippet_len

        rec_start_end_idx_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=len(rec_dict['spectrogram']),
                snippet_len=snippet_len
            )
            rec_dict['split_list'] = split_list
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list.extend(t)

        return rec_start_end_idx_list

    def map_idx_to_data_fn(self, idx):

        def py_fn(idx):

            idx = idx.numpy().item()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = self.np_dataset[rec_idx]
            snippet_len = self.model.config.snippet_len
            assert start_frame % snippet_len == 0
            snippet_idx = start_frame // snippet_len
            spec = rec_dict['spectrogram'][start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            return rec_idx, snippet_idx, spec, notes

        rec_idx, snippet_idx, spec, notes = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.float32, tf.float32]
        )
        rec_idx.set_shape([])
        snippet_idx.set_shape([])
        spec.set_shape([None, 500])
        notes.set_shape([None])

        return dict(
            rec_idx=rec_idx,
            snippet_idx=snippet_idx,
            spectrogram=spec,
            notes=notes
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(10)

        return dataset


class TFDatasetForRWC(TFDataset):

    def __init__(self, model):

        super(TFDatasetForRWC, self).__init__(model)
        is_inferencing = model.config.train_or_inference.inference is not None
        assert is_inferencing

        self.rec_files = TFDatasetForRWC.get_rec_files_fn()
        self.np_dataset = self.gen_np_dataset_fn()
        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)
        num_frames = [len(rec_dict['spectrogram']) for rec_dict in self.np_dataset]
        self.num_frames_vector = np.asarray(num_frames, dtype=np.int64)
        self.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_split_list_fn(self.np_dataset)
        self.tf_dataset = self.gen_tf_dataset_fn()

    @staticmethod
    def get_rec_files_fn():

        dir_prefix = 'RWC-MDB-P-2001-M0'
        dir_path = os.path.join(os.environ['rwc'], 'popular', dir_prefix)
        num_recordings = []
        for dir_idx in range(7):
            dir_idx = dir_idx + 1
            disk_dir = dir_path + str(dir_idx)
            aiff_files = glob.glob(os.path.join(disk_dir, '*.aiff'))
            num_recordings.append(len(aiff_files))
        start_end_list = np.cumsum(num_recordings)
        assert start_end_list[-1] == 100
        start_end_list = np.pad(start_end_list, [[1, 0]])

        rec_files = []
        for rec_idx in range(100):
            disk_idx = np.searchsorted(start_end_list, rec_idx, side='right')
            assert disk_idx >= 1
            disk_idx = disk_idx - 1
            disk_path = dir_path + str(disk_idx + 1)
            disk_start_rec_idx = start_end_list[disk_idx]
            rec_idx_within_disk = rec_idx - disk_start_rec_idx
            rec_idx_within_disk = rec_idx_within_disk + 1

            recs = glob.glob(os.path.join(disk_path, '*.aiff'))
            assert len(recs) == num_recordings[disk_idx]
            for recording_name in recs:
                t = os.path.basename(recording_name)
                t = t.split()[0]
                if t == str(rec_idx_within_disk):
                    rec_files.append(recording_name)
                    break
            else:
                assert False
        assert len(rec_files) == 100
        t = set(rec_files)
        assert len(t) == 100

        return rec_files

    def get_num_frames_fn(self, track_id):

        assert isinstance(track_id, str)
        rec_idx = int(track_id)
        assert rec_idx >= 0
        aiff_file = self.rec_files[rec_idx]
        aiff_info = soundfile.info(aiff_file)
        assert aiff_info.samplerate == 44100
        num_samples = aiff_info.frames
        h = 441
        num_frames = (num_samples + h - 1) // h

        return num_frames

    def gen_spec_fn(self, track_id):

        assert isinstance(track_id, str)
        rec_idx = int(track_id)
        nsgt_instances = TFDataset.nsgt_instances
        nsgt_Lses = TFDataset.nsgt_Lses
        hop_size = TFDataset.nsgt_hop_size
        num_freq_bins = TFDataset.nsgt_freq_bins
        sr = 44100

        mix_wav_file = self.rec_files[rec_idx]
        samples, _sr = librosa.load(mix_wav_file, sr=None, dtype=np.float64)
        assert _sr == sr
        assert samples.ndim == 1
        assert samples.max() < 1
        assert samples.min() >= -1
        samples = samples.astype(np.float32)
        num_samples = len(samples)
        t = np.searchsorted(nsgt_Lses, num_samples)
        assert 1 <= t <= len(nsgt_instances)
        nsgt_fn = nsgt_instances[t - 1].nsgt_of_wav_file_fn
        nsgt = nsgt_fn(samples)
        num_frames = (num_samples + hop_size - 1) // hop_size
        assert nsgt.shape == (num_frames, num_freq_bins)
        nsgt = nsgt[::4, 1:501]
        nsgt = TFDataset.spec_transform_fn(nsgt)
        nsgt = np.require(nsgt, np.float32, requirements=['O', 'C'])

        return nsgt

    def load_melody_from_file_fn(self, track_id):

        assert isinstance(track_id, str)
        rec_idx = int(track_id)
        melody_dir = os.path.join(os.environ['rwc'], 'popular', 'AIST.RWC-MDB-P-2001.MELODY')
        melody_prefix = 'RM-P'
        melody_suffix = '.MELODY.TXT'
        melody_file = melody_prefix + str(rec_idx + 1).zfill(3) + melody_suffix
        melody_file = os.path.join(melody_dir, melody_file)

        with open(melody_file, 'r') as fh:
            lines = fh.readlines()
            line = lines[-1]
            cols = line.split()
            num_frames = int(cols[0]) + 1
            aiff_num_frames = self.get_num_frames_fn(track_id)
            assert num_frames <= aiff_num_frames
            freqs = np.zeros([aiff_num_frames], np.float32)
            min_freq = 31.
            for line in lines:
                cols = line.split()
                assert len(cols) == 5
                assert cols[0] == cols[1]
                assert cols[2] == 'm'
                frame_idx = int(cols[0])
                assert frame_idx >= 0
                freq = float(cols[3])
                assert freq == 0 or freq > min_freq
                freqs[frame_idx] = freq

            return freqs

    def gen_label_fn(self, track_id):

        freqs_441 = self.load_melody_from_file_fn(track_id)

        num_frames_441 = len(freqs_441)
        num_frames_256 = 1 + ((num_frames_441 - 1) * 441 + 255) // 256
        times_441 = np.arange(num_frames_441) * 0.01
        times_256 = np.arange(num_frames_256) * (256. / 44100.)
        assert times_256[-1] >= times_441[-1]
        TFDataset.validity_check_of_ref_freqs_fn(freqs_441)
        voicing_441 = freqs_441 > .1
        freqs_256, _ = mir_eval.melody.resample_melody_series(
            times=times_441,
            frequencies=freqs_441,
            voicing=voicing_441,
            times_new=times_256
        )
        TFDataset.validity_check_of_ref_freqs_fn(freqs_256)
        notes = TFDataset.hz_to_midi_fn(freqs_256)

        result = dict(notes=notes, original=dict(times=times_441, freqs=freqs_441))

        return result

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - rwc - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.info('{}/{} - {}'.format(track_idx + 1, num_tracks, track_id))

            spec = self.gen_spec_fn(track_id)
            notes_original_dict = self.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - len(spec)
            assert np.abs(diff) <= 1
            if diff == 1:
                spec = np.pad(spec, [[0, 1], [0, 0]])
            elif diff == -1:
                notes = np.pad(notes, [[0, 1]])
            assert len(notes) == len(spec)

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        return dataset

    def gen_split_list_fn(self, np_dataset):

        model = self.model
        snippet_len = model.config.snippet_len

        rec_start_end_idx_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=len(rec_dict['spectrogram']),
                snippet_len=snippet_len
            )
            rec_dict['split_list'] = split_list
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list.extend(t)

        return rec_start_end_idx_list

    def map_idx_to_data_fn(self, idx):

        def py_fn(idx):

            idx = idx.numpy().item()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = self.np_dataset[rec_idx]
            snippet_len = self.model.config.snippet_len
            assert start_frame % snippet_len == 0
            snippet_idx = start_frame // snippet_len
            spec = rec_dict['spectrogram'][start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            return rec_idx, snippet_idx, spec, notes

        rec_idx, snippet_idx, spec, notes = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.float32, tf.float32]
        )
        rec_idx.set_shape([])
        snippet_idx.set_shape([])
        spec.set_shape([None, 500])
        notes.set_shape([None])

        return dict(
            rec_idx=rec_idx,
            snippet_idx=snippet_idx,
            spectrogram=spec,
            notes=notes
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(10)

        return dataset


class MetricsTrainingModeTrainingSplit:

    def __init__(self, model):

        assert model.config.train_or_inference.inference is None
        assert model.is_training
        assert 'train' in model.name

        self.model = model
        self.voicing_threshold = model.config.acoustic_model_ins.voicing_threshold
        self.var_dict = self.define_tf_variables_fn()

        self.oa = None
        self.loss = None
        self.current_voicing_threshold = None

    def reset(self):

        for var in self.var_dict['all_updated']:
            var = var.deref()
            var.assign(tf.zeros_like(var))

        self.oa = None
        self.loss = None
        self.current_voicing_threshold = None

    def define_tf_variables_fn(self):

        model = self.model

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                all_defined_vars_updated = dict()

                def gen_tf_var(name):

                    assert name

                    with tf.device('/cpu:0'):
                        var = tf.Variable(
                            initial_value=tf.zeros([], dtype=tf.int64),
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

                batch_counter = tf.Variable(
                    initial_value=tf.zeros([], dtype=tf.int32),
                    trainable=False,
                    name='batch_counter'
                )
                all_defined_vars_updated[batch_counter.ref()] = False
                loss = tf.Variable(
                    initial_value=tf.zeros([], tf.float32),
                    trainable=False,
                    name='acc_loss'
                )
                all_defined_vars_updated[loss.ref()] = False

                return dict(
                    melody=melody_var_dict,
                    batch_counter=batch_counter,
                    loss=loss,
                    all_updated=all_defined_vars_updated
                )

    def update_melody_var_fn(self, l1, l2, value):

        assert l1 is not None
        assert l2 is not None
        assert value is not None

        var_dict = self.var_dict['melody']
        all_updated = self.var_dict['all_updated']
        var = var_dict[l1][l2]

        var_ref = var.ref()
        assert not all_updated[var_ref]

        var.assign_add(value)
        all_updated[var_ref] = True

    def update_loss_fn(self, value):

        assert value is not None

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['loss']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        assert value.dtype == var.dtype
        var.assign_add(value)
        all_updated[var_ref] = True

    def increase_batch_counter_fn(self):

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['batch_counter']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        var.assign_add(1)
        all_updated[var_ref] = True

    @tf.function(input_signature=[
        tf.TensorSpec([None]),
        tf.TensorSpec([None, 320]),
        tf.TensorSpec([])
    ], autograph=False)
    def update_states_tf_fn(self, ref_notes, logits, loss):

        voicing_threshold = self.voicing_threshold
        count_nonzero_fn = MetricsBase.count_nonzero_fn

        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        loss = tf.convert_to_tensor(loss, tf.float32)

        est_probs = tf.sigmoid(logits)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_peak_indices = tf.argmax(est_probs, axis=1, output_type=tf.int32)
        est_peak_probs = tf.gather(est_probs, indices=est_peak_indices, axis=1, batch_dims=1)
        est_voicing = est_peak_probs > voicing_threshold
        est_voicing.set_shape([None])
        n_est_voicing = tf.logical_not(est_voicing)

        est_notes = MetricsBase.est_notes_fn(est_peak_indices=est_peak_indices, est_probs=est_probs)

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
        self.update_melody_var_fn('gt', 'voiced', voiced_frames)
        self.update_melody_var_fn('gt', 'unvoiced', unvoiced_frames)
        self.update_melody_var_fn('voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn('voicing', 'incorrect_voiced', incorrect_voiced_frames)
        self.update_melody_var_fn('voicing', 'correct_unvoiced', correct_unvoiced_frames)

        correct_pitches_wide = est_ref_note_diffs < .5
        correct_pitches_wide = tf.logical_and(ref_voicing, correct_pitches_wide)
        correct_pitches_strict = tf.logical_and(est_voicing, correct_pitches_wide)
        correct_pitches_wide = count_nonzero_fn(correct_pitches_wide)
        correct_pitches_strict = count_nonzero_fn(correct_pitches_strict)
        self.update_melody_var_fn('correct_pitches', 'wide', correct_pitches_wide)
        self.update_melody_var_fn('correct_pitches', 'strict', correct_pitches_strict)

        correct_chromas_wide = est_ref_note_diffs
        octave = MetricsBase.octave(correct_chromas_wide)
        correct_chromas_wide = tf.abs(correct_chromas_wide - octave) < .5
        correct_chromas_wide = tf.logical_and(ref_voicing, correct_chromas_wide)
        correct_chromas_strict = tf.logical_and(est_voicing, correct_chromas_wide)
        correct_chromas_wide = count_nonzero_fn(correct_chromas_wide)
        correct_chromas_strict = count_nonzero_fn(correct_chromas_strict)
        self.update_melody_var_fn('correct_chromas', 'wide', correct_chromas_wide)
        self.update_melody_var_fn('correct_chromas', 'strict', correct_chromas_strict)

        self.update_loss_fn(loss)
        self.increase_batch_counter_fn()

        assert all(self.var_dict['all_updated'].values())

    def update_states(self, ref_notes, logits, loss):

        t = [ref_notes, logits, loss]
        assert all(isinstance(v, tf.Tensor) for v in t)

        self.update_states_tf_fn(ref_notes, logits, loss)

    def results(self):

        melody_dict = self.var_dict['melody']
        var_loss = self.var_dict['loss']
        var_batch_counter = self.var_dict['batch_counter']
        f8f4div = MetricsBase.to_f8_divide_and_to_f4_fn

        correct_frames = melody_dict['correct_pitches']['strict'] + melody_dict['voicing']['correct_unvoiced']
        num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        m_oa = f8f4div(correct_frames, num_frames_vector)
        m_oa.set_shape([])

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
        m_loss = var_loss / tf.cast(var_batch_counter, tf.float32)

        self.oa = m_oa.numpy().item()
        self.loss = m_loss.numpy().item()
        self.current_voicing_threshold = self.voicing_threshold.read_value().numpy()

        results = dict(
            loss=m_loss,
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


class MetricsBase:

    """
    base metric class for validation split in training mode and inference mode
    """

    def __init__(self, model):

        self.model = model
        self.num_recs = len(model.config.tvt_split_dict[model.name])

    def update_melody_var_fn(self, rec_idx, l1, l2, value, viterbi=False):

        assert rec_idx is not None
        assert l1 is not None
        assert l2 is not None
        assert value is not None

        if not viterbi:
            var_dict = self.var_dict['melody']
            all_updated = self.var_dict['all_updated']
        else:
            var_dict = self.viterbi_var_dict['melody']
            all_updated = self.viterbi_var_dict['all_updated']

        var = var_dict[l1][l2]

        var_ref = var.ref()
        assert not all_updated[var_ref]

        with tf.device(var.device):
            value = tf.identity(value)
        var.scatter_add(tf.IndexedSlices(values=value, indices=rec_idx))
        all_updated[var_ref] = True

    def update_loss_fn(self, value):

        assert value is not None

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['loss']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        assert value.dtype == var.dtype
        var.assign_add(value)
        all_updated[var_ref] = True

    def increase_batch_counter_fn(self):

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['batch_counter']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        var.assign_add(1)
        all_updated[var_ref] = True

    @staticmethod
    def to_f8_divide_and_to_f4_fn(numerator, denominator):

        numerator = tf.cast(numerator, tf.float64)
        denominator = tf.cast(denominator, tf.float64)
        numerator = numerator / tf.maximum(denominator, 1e-7)
        numerator = tf.cast(numerator, tf.float32)

        return numerator

    @staticmethod
    def count_nonzero_fn(inputs):

        outputs = inputs
        outputs = tf.math.count_nonzero(outputs, dtype=tf.int32)
        outputs = tf.cast(outputs, tf.int64)

        return outputs

    @staticmethod
    def est_notes_fn(est_peak_indices, est_probs):

        note_range = TFDataset.note_range
        note_offset = note_range[0]
        assert np.isclose(note_offset, 23.6)
        note_range = note_range - note_offset
        note_range = tf.constant(note_range, dtype=tf.float32)

        frames_320 = tf.range(320, dtype=tf.int32)
        peak_masks = est_peak_indices[:, None] - frames_320[None, :]
        peak_masks.set_shape([None, 320])
        peak_masks = tf.abs(peak_masks) <= 1
        masked_probs = tf.where(peak_masks, est_probs, tf.zeros_like(est_probs))
        masked_probs.set_shape([None, 320])
        normalization_probs = tf.reduce_sum(masked_probs, axis=1)
        normalization_probs.set_shape([None])
        frames_64 = note_range
        est_notes = frames_64[None, :] * masked_probs
        est_notes = tf.reduce_sum(est_notes, axis=1)
        est_notes.set_shape([None])
        est_notes = est_notes / tf.maximum(normalization_probs, 1e-3)
        est_notes = est_notes + note_offset

        return est_notes

    @staticmethod
    def octave(distance):

        distance = tf.floor(distance / 12. + .5) * 12.

        return distance


class MetricsValidation(MetricsBase):

    def __init__(self, model):

        super(MetricsValidation, self).__init__(model)

        self.current_voicing_threshold = None
        self.oa = None
        self.loss = None
        self.rec_idx = None
        self.snippet_idx = None

        assert model.config.train_or_inference.inference is None
        assert 'validation' in model.name
        assert not model.is_training

        t = np.arange(.01, 1., .01, dtype=np.float64)
        t = t.astype(np.float32)
        self.voicing_thresholds = t
        self.num_voicing_thresholds = len(self.voicing_thresholds)

        self.var_dict = self.define_tf_variables_fn()
        self.num_frames_vector = tf.convert_to_tensor(model.tf_dataset.num_frames_vector, tf.int64)

    def reset(self):

        for var in self.var_dict['all_updated']:
            var = var.deref()
            var.assign(tf.zeros_like(var))

        self.oa = None
        self.loss = None
        self.rec_idx = None
        self.snippet_idx = None
        self.current_voicing_threshold = None

    def define_tf_variables_fn(self):

        model = self.model
        num_recs = self.num_recs
        num_ths = self.num_voicing_thresholds

        with tf.name_scope(model.name):
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
                    ),
                    correct_pitches=dict(
                        wide=gen_tf_var('correct_pitches_wide', [num_recs]),
                        strict=gen_tf_var('correct_pitches_strict', [num_recs, num_ths])
                    ),
                    correct_chromas=dict(
                        wide=gen_tf_var('correct_chromas_wide', [num_recs]),
                        strict=gen_tf_var('correct_chromas_strict', [num_recs, num_ths])
                    )
                )

                batch_counter = tf.Variable(
                    initial_value=tf.zeros([], tf.int32),
                    trainable=False,
                    name='batch_counter'
                )
                all_defined_vars_updated[batch_counter.ref()] = False
                loss = tf.Variable(
                    initial_value=tf.zeros([], tf.float32),
                    trainable=False,
                    name='acc_loss'
                )
                all_defined_vars_updated[loss.ref()] = False

                return dict(
                    melody=melody_var_dict,
                    batch_counter=batch_counter,
                    loss=loss,
                    all_updated=all_defined_vars_updated
                )

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([None]),
        tf.TensorSpec([None, 320]),
        tf.TensorSpec([])
    ], autograph=False)
    def update_states_tf_fn(self, rec_idx, ref_notes, logits, loss):

        num_voicing_thresholds = self.num_voicing_thresholds
        voicing_thresholds = tf.convert_to_tensor(self.voicing_thresholds, tf.float32)
        count_nonzero_fn = self.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        loss = tf.convert_to_tensor(loss, tf.float32)

        est_probs = tf.sigmoid(logits)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_peak_indices = tf.argmax(est_probs, axis=1, output_type=tf.int32)
        est_peak_probs = tf.gather(est_probs, indices=est_peak_indices, axis=1, batch_dims=1)
        est_voicing = est_peak_probs[:, None] > voicing_thresholds[None, :]
        est_voicing.set_shape([None, num_voicing_thresholds])
        n_est_voicing = tf.logical_not(est_voicing)

        est_notes = self.est_notes_fn(est_peak_indices=est_peak_indices, est_probs=est_probs)
        est_ref_note_diffs = tf.abs(est_notes - ref_notes)
        est_ref_note_diffs.set_shape([None])

        def count_nonzero_for_multi_ths_fn(inputs):

            inputs.set_shape([None, num_voicing_thresholds])
            outputs = tf.math.count_nonzero(inputs, axis=0, dtype=tf.int32)
            outputs = tf.cast(outputs, tf.int64)

            return outputs

        voiced_frames = count_nonzero_fn(ref_voicing)
        unvoiced_frames = tf.size(ref_voicing, tf.int64) - voiced_frames
        ref_voicing_logical_and_est_voicing = tf.logical_and(ref_voicing[:, None], est_voicing)
        ref_voicing_logical_and_est_voicing.set_shape([None, num_voicing_thresholds])
        correct_voiced_frames = count_nonzero_for_multi_ths_fn(ref_voicing_logical_and_est_voicing)
        correct_voiced_frames.set_shape([num_voicing_thresholds])
        incorrect_voiced_frames = tf.logical_and(n_ref_voicing[:, None], est_voicing)
        incorrect_voiced_frames = count_nonzero_for_multi_ths_fn(incorrect_voiced_frames)
        correct_unvoiced_frames = tf.logical_and(n_ref_voicing[:, None], n_est_voicing)
        correct_unvoiced_frames = count_nonzero_for_multi_ths_fn(correct_unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'gt', 'voiced', voiced_frames)
        self.update_melody_var_fn(rec_idx, 'gt', 'unvoiced', unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'incorrect_voiced', incorrect_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_unvoiced', correct_unvoiced_frames)

        correct_pitches_wide = est_ref_note_diffs < .5
        correct_pitches_wide = tf.logical_and(ref_voicing, correct_pitches_wide)
        correct_pitches_wide.set_shape([None])
        correct_pitches_strict = tf.logical_and(correct_pitches_wide[:, None], est_voicing)
        correct_pitches_strict.set_shape([None, num_voicing_thresholds])
        correct_pitches_wide = count_nonzero_fn(correct_pitches_wide)
        correct_pitches_strict = count_nonzero_for_multi_ths_fn(correct_pitches_strict)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'wide', correct_pitches_wide)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'strict', correct_pitches_strict)

        correct_chromas_wide = est_ref_note_diffs
        octave = self.octave(correct_chromas_wide)
        correct_chromas_wide = tf.abs(correct_chromas_wide - octave) < .5
        correct_chromas_wide = tf.logical_and(ref_voicing, correct_chromas_wide)
        correct_chromas_wide.set_shape([None])
        correct_chromas_strict = tf.logical_and(est_voicing, correct_chromas_wide[:, None])
        correct_chromas_strict.set_shape([None, num_voicing_thresholds])
        correct_chromas_wide = count_nonzero_fn(correct_chromas_wide)
        correct_chromas_strict = count_nonzero_for_multi_ths_fn(correct_chromas_strict)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'wide', correct_chromas_wide)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'strict', correct_chromas_strict)

        self.update_loss_fn(loss)
        self.increase_batch_counter_fn()

        assert all(self.var_dict['all_updated'].values())

    def update_states(self, rec_idx, snippet_idx, ref_notes, logits, loss):

        model = self.model

        t = [rec_idx, snippet_idx, ref_notes, logits, loss]
        assert all(isinstance(v, tf.Tensor) for v in t)

        _rec_idx = rec_idx.numpy()
        rec_dict = model.tf_dataset.np_dataset[_rec_idx]
        num_snippets = len(rec_dict['split_list'])
        _snippet_idx = snippet_idx.numpy()
        assert _snippet_idx < num_snippets

        if _snippet_idx == 0:
            self.rec_idx = _rec_idx
            self.snippet_idx = _snippet_idx

        assert _rec_idx == self.rec_idx

        if _snippet_idx > 0:
            assert _snippet_idx == self.snippet_idx + 1

        self.snippet_idx = _snippet_idx

        self.update_states_tf_fn(
            rec_idx=rec_idx,
            ref_notes=ref_notes,
            logits=logits,
            loss=loss
        )

    def best_voicing_threshold_fn(self):

        model = self.model
        melody_dict = self.var_dict['melody']
        f8f4div = self.to_f8_divide_and_to_f4_fn
        num_recs = self.num_recs
        num_ths = self.num_voicing_thresholds
        num_frames_vector = self.num_frames_vector

        _num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        tf.debugging.assert_equal(_num_frames_vector, num_frames_vector)
        va = f8f4div(
            numerator=melody_dict['voicing']['correct_voiced'] + melody_dict['voicing']['correct_unvoiced'],
            denominator=num_frames_vector[:, None]
        )
        va.set_shape([num_recs, num_ths])
        va = tf.reduce_mean(va, axis=0)
        best_th_idx = tf.argmax(va, output_type=tf.int32).numpy()
        best_th = self.voicing_thresholds[best_th_idx]
        var_voicing_threshold = model.config.acoustic_model_ins.voicing_threshold
        assert isinstance(var_voicing_threshold, tf.Variable)
        _best_th = var_voicing_threshold.assign(best_th)
        _best_th = _best_th.numpy()
        assert np.isclose(best_th, _best_th)
        logging.debug('voicing threshold set to {}'.format(_best_th))

        self.current_voicing_threshold = best_th

        return best_th_idx

    def results(self):

        melody_dict = self.var_dict['melody']
        var_loss = self.var_dict['loss']
        var_batch_counter = self.var_dict['batch_counter']
        f8f4div = self.to_f8_divide_and_to_f4_fn
        num_frames_vector = self.num_frames_vector

        best_th_idx = self.best_voicing_threshold_fn()

        m_vrr = f8f4div(
            numerator=melody_dict['voicing']['correct_voiced'][:, best_th_idx],
            denominator=melody_dict['gt']['voiced']
        )
        m_vfa = f8f4div(
            numerator=melody_dict['voicing']['incorrect_voiced'][:, best_th_idx],
            denominator=melody_dict['gt']['unvoiced']
        )
        m_va = f8f4div(
            numerator=(melody_dict['voicing']['correct_voiced'][:, best_th_idx] +
                       melody_dict['voicing']['correct_unvoiced'][:, best_th_idx]),
            denominator=num_frames_vector
        )
        m_rpa_strict = f8f4div(
            numerator=melody_dict['correct_pitches']['strict'][:, best_th_idx],
            denominator=melody_dict['gt']['voiced']
        )
        m_rpa_wide = f8f4div(
            numerator=melody_dict['correct_pitches']['wide'],
            denominator=melody_dict['gt']['voiced']
        )
        m_rca_strict = f8f4div(
            numerator=melody_dict['correct_chromas']['strict'][:, best_th_idx],
            denominator=melody_dict['gt']['voiced']
        )
        m_rca_wide = f8f4div(
            numerator=melody_dict['correct_chromas']['wide'],
            denominator=melody_dict['gt']['voiced']
        )
        m_oa = f8f4div(
            numerator=(melody_dict['correct_pitches']['strict'][:, best_th_idx] +
                       melody_dict['voicing']['correct_unvoiced'][:, best_th_idx]),
            denominator=num_frames_vector
        )
        m_loss = var_loss / tf.cast(var_batch_counter, tf.float32)

        self.oa = tf.reduce_mean(m_oa).numpy()
        self.loss = m_loss.numpy()

        results = dict(
            loss=m_loss,
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


class Viterbi:

    def __init__(self):

        self.num_freq_bins = 320
        self.h = 256
        self.sr = 44100
        self.single_side_peak_width = 5

        th = 0.31
        self.threshold = np.log(th / (1. - th))
        self.transition_matrix = self.transition_matrix_fn()

        self.ini_probs = Viterbi.init_probs_fn()

    def find_peaks_fn(self, aaa):

        spw = self.single_side_peak_width
        bbb = np.pad(aaa, [(spw, spw)], mode='reflect')
        bbb = librosa.util.frame(bbb, frame_length=2 * spw + 1, hop_length=1)
        bbb = np.argmax(bbb, axis=0)
        bbb = bbb == spw

        return bbb

    @tf.function(
        input_signature=[tf.TensorSpec([None, 320])],
        autograph=False
    )
    def find_peaks_all_at_once_tf_fn(self, frames_logits):

        frames_logits = tf.convert_to_tensor(frames_logits, np.float32)
        frames_logits.set_shape([None, 320])
        spw = self.single_side_peak_width
        frames_logits = tf.pad(frames_logits, [(0, 0), (spw, spw)], mode='reflect')
        frames_logits = tf.signal.frame(frames_logits, frame_length=2 * spw + 1, frame_step=1, axis=-1)
        frames_logits.set_shape([None, 320, 2 * spw + 1])
        frames_logits = tf.argmax(frames_logits, axis=-1, output_type=tf.int32)
        frames_logits.set_shape([None, 320])
        frames_logits = frames_logits == spw

        return frames_logits

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
        frames_logits_are_peaks = self.find_peaks_all_at_once_tf_fn(frames_logits).numpy()
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
                p_voiced = np.exp(scale * (global_peak_logit - threshold) + offset)
            else:
                p_voiced = np.exp(scale * (global_peak_logit - threshold) - offset)
            p_voiced = p_voiced / (1. + p_voiced)

            np.exp(peak_logits, out=peak_logits)
            t = p_voiced / np.sum(peak_logits)
            np.multiply(peak_logits, t, out=peak_logits)
            melodies_frames[peak_indices, frame_idx] = peak_logits
            melodies_frames[-1, frame_idx] = 1. - p_voiced

        t = np.sum(melodies_frames, axis=0)
        assert np.all(np.isclose(t, 1))

        return melodies_frames

    @staticmethod
    def thresholds_and_probs_fn():

        voicing_threshold = 0.31
        min_th = 0.25
        thresholds = np.linspace(min_th, voicing_threshold, 51).astype(np.float32)
        sigma = (voicing_threshold - min_th) / 3.
        probs = np.exp(-(thresholds - voicing_threshold) ** 2 / (2. * sigma ** 2))
        probs = probs / np.sum(probs)
        thresholds = np.log(thresholds / (1. - thresholds))

        return thresholds, probs

    @staticmethod
    def init_probs_fn():

        file_name = 'viterbi_init_probs.dat'
        _name, probs = load_np_array_from_file_fn(file_name)
        assert _name == file_name[:-4]
        assert np.sum(probs) == 1.
        assert np.all(probs > 0)

        return probs

    def transition_matrix_fn(self):

        # h = self.h
        # sr = self.sr
        # B = self.B
        # n_bins = self.num_freq_bins
        #
        # bins_per_hop = self.transition_max_octs_per_s * float(h) / sr * B
        # bins_per_hop = int(bins_per_hop)
        # if bins_per_hop % 2 == 0:
        #     bins_per_hop += 1
        #
        # voiced_transition = librosa.sequence.transition_local(
        #     n_states=n_bins,
        #     width=bins_per_hop,
        #     window='triangle',
        #     wrap=False
        # )
        # change = self.prob_changing_voiced_state
        # nchange = 1. - change
        #
        # transition_matrix = np.empty([n_bins + 1, n_bins + 1], np.float32)
        # transition_matrix[:n_bins, :n_bins] = nchange * voiced_transition
        # transition_matrix[:n_bins, -1] = change
        # transition_matrix[-1, :n_bins] = change / float(n_bins)
        # transition_matrix[-1, -1] = nchange

        name, transition_matrix = load_np_array_from_file_fn('viterbi_transition_matrix.dat')
        assert name == 'viterbi_transition_matrix'
        t = np.sum(transition_matrix, axis=1)
        assert np.all(np.isclose(t, 1))

        return transition_matrix

    def __call__(self, logits):

        observation_probs = self.observation_probs_fn(logits)
        bins = Viterbi.viterbi_librosa_fn(
            transition_matrix=self.transition_matrix,
            prob_init=self.ini_probs,
            probs_st=observation_probs
        )
        n_bins = self.num_freq_bins
        voiced = bins < n_bins
        bins = np.minimum(bins, n_bins - 1)

        return voiced, bins

    @staticmethod
    def viterbi_librosa_fn(*, transition_matrix, prob_init, probs_st):

        """
        This has been optimized by shaun.
        Args:
            B: S * S, sum(B, axis=1) = 1
            prob_init: S, sum(S) = 1
            probs: S * T, prob(observation at t| in state s at t)

        Returns:

        """

        B = transition_matrix
        probs = probs_st

        S = len(B)
        T = probs.shape[1]
        assert B.shape == (S, S)
        assert probs.shape == (S, T)
        t = np.sum(B, axis=1)
        assert np.allclose(t, 1.)
        assert len(prob_init) == S
        assert np.isclose(np.sum(prob_init), 1.)

        tinyp = np.finfo(probs.dtype).tiny

        B = np.log(B.T + tinyp)  # S * S
        B = np.require(B, requirements=['C'])
        prob_init = np.log(prob_init + tinyp)
        probs = np.log(probs.T + tinyp)  # T * S
        probs = np.require(probs, requirements=['C'])

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


class MetricsInference(MetricsBase):

    viterbi = Viterbi()

    def __init__(self, model):

        super(MetricsInference, self).__init__(model)

        self.oa = None
        self.loss = None
        self.rec_idx = None
        self.snippet_idx = None
        self.mir_eval_oas = []
        self.tf_oas = None

        self.viterbi_tf_oas = None
        self.viterbi_mir_eval_oas = None

        assert model.config.train_or_inference.inference is not None

        self.voicing_threshold = model.config.acoustic_model_ins.voicing_threshold
        assert isinstance(self.voicing_threshold, tf.Variable)

        self.var_dict = self.define_tf_variables_fn()

        self.viterbi_var_dict = self.viterbi_define_tf_variables_fn()

    def define_tf_variables_fn(self):

        model = self.model
        num_recs = self.num_recs

        with tf.name_scope(model.name):
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

                batch_counter = tf.Variable(
                    initial_value=tf.zeros([], dtype=tf.int32),
                    trainable=False,
                    name='batch_counter'
                )
                all_defined_vars_updated[batch_counter.ref()] = False
                loss = tf.Variable(
                    initial_value=tf.zeros([], tf.float32),
                    trainable=False,
                    name='acc_loss'
                )
                all_defined_vars_updated[loss.ref()] = False

                return dict(
                    melody=melody_var_dict,
                    batch_counter=batch_counter,
                    loss=loss,
                    all_updated=all_defined_vars_updated
                )

    def viterbi_define_tf_variables_fn(self):

        model = self.model
        num_recs = self.num_recs

        with tf.name_scope('viterbi'):
            with tf.name_scope(model.name):
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

    def reset(self):

        for var in self.var_dict['all_updated']:
            var = var.deref()
            var.assign(tf.zeros_like(var))

        self.oa = None
        self.loss = None
        self.rec_idx = None
        self.snippet_idx = None
        self.mir_eval_oas = []
        self.tf_oas = None

        for var in self.viterbi_var_dict['all_updated']:
            var = var.deref()
            var.assign(tf.zeros_like(var))

        self.viterbi_tf_oas = None
        self.viterbi_mir_eval_oas = []

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([None]),
        tf.TensorSpec([None, 320]),
        tf.TensorSpec([])
    ], autograph=False)
    def update_states_tf_fn(self, rec_idx, ref_notes, logits, loss):

        voicing_threshold = self.voicing_threshold
        assert isinstance(voicing_threshold, tf.Variable)
        count_nonzero_fn = self.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        loss = tf.convert_to_tensor(loss, tf.float32)

        est_probs = tf.sigmoid(logits)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_peak_indices = tf.argmax(est_probs, axis=1, output_type=tf.int32)
        est_peak_probs = tf.gather(est_probs, indices=est_peak_indices, axis=1, batch_dims=1)
        est_voicing = est_peak_probs > voicing_threshold
        est_voicing.set_shape([None])
        n_est_voicing = tf.logical_not(est_voicing)

        est_notes = self.est_notes_fn(est_peak_indices=est_peak_indices, est_probs=est_probs)

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
        self.update_melody_var_fn(rec_idx, 'gt', 'unvoiced', unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'incorrect_voiced', incorrect_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_unvoiced', correct_unvoiced_frames)

        correct_pitches_wide = est_ref_note_diffs < .5
        correct_pitches_wide = tf.logical_and(ref_voicing, correct_pitches_wide)
        correct_pitches_strict = tf.logical_and(est_voicing, correct_pitches_wide)
        correct_pitches_wide = count_nonzero_fn(correct_pitches_wide)
        correct_pitches_strict = count_nonzero_fn(correct_pitches_strict)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'wide', correct_pitches_wide)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'strict', correct_pitches_strict)

        correct_chromas_wide = est_ref_note_diffs
        octave = self.octave(correct_chromas_wide)
        correct_chromas_wide = tf.abs(correct_chromas_wide - octave) < .5
        correct_chromas_wide = tf.logical_and(ref_voicing, correct_chromas_wide)
        correct_chromas_strict = tf.logical_and(est_voicing, correct_chromas_wide)
        correct_chromas_wide = count_nonzero_fn(correct_chromas_wide)
        correct_chromas_strict = count_nonzero_fn(correct_chromas_strict)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'wide', correct_chromas_wide)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'strict', correct_chromas_strict)

        self.update_loss_fn(loss)
        self.increase_batch_counter_fn()

        assert all(self.var_dict['all_updated'].values())

        est_notes_with_voicing_info = tf.where(est_voicing, est_notes, -est_notes)

        return est_notes_with_voicing_info

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.int32, name='rec_idx'),
        tf.TensorSpec([None], name='ref_notes'),
        tf.TensorSpec([None, 320], name='logits'),
        tf.TensorSpec([None], tf.int32, name='melody_bins'),
        tf.TensorSpec([None], tf.bool, name='est_voicing')
    ], autograph=False)
    def viterbi_update_states_tf_fn(self, rec_idx, ref_notes, logits, melody_bins, est_voicing):

        count_nonzero_fn = MetricsBase.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)
        melody_bins = tf.convert_to_tensor(melody_bins, tf.int32)
        est_voicing = tf.convert_to_tensor(est_voicing, tf.bool)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        n_est_voicing = tf.logical_not(est_voicing)

        est_notes = MetricsBase.est_notes_fn(est_peak_indices=melody_bins, est_probs=tf.sigmoid(logits))

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
        self.update_melody_var_fn(rec_idx, 'gt', 'voiced', voiced_frames, viterbi=True)
        self.update_melody_var_fn(rec_idx, 'gt', 'unvoiced', unvoiced_frames, viterbi=True)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_voiced', correct_voiced_frames, viterbi=True)
        self.update_melody_var_fn(rec_idx, 'voicing', 'incorrect_voiced', incorrect_voiced_frames, viterbi=True)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_unvoiced', correct_unvoiced_frames, viterbi=True)

        correct_pitches_wide = est_ref_note_diffs < .5
        correct_pitches_wide = tf.logical_and(ref_voicing, correct_pitches_wide)
        correct_pitches_strict = tf.logical_and(est_voicing, correct_pitches_wide)
        correct_pitches_wide = count_nonzero_fn(correct_pitches_wide)
        correct_pitches_strict = count_nonzero_fn(correct_pitches_strict)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'wide', correct_pitches_wide, viterbi=True)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', 'strict', correct_pitches_strict, viterbi=True)

        correct_chromas_wide = est_ref_note_diffs
        octave = MetricsBase.octave(correct_chromas_wide)
        correct_chromas_wide = tf.abs(correct_chromas_wide - octave) < .5
        correct_chromas_wide = tf.logical_and(ref_voicing, correct_chromas_wide)
        correct_chromas_strict = tf.logical_and(est_voicing, correct_chromas_wide)
        correct_chromas_wide = count_nonzero_fn(correct_chromas_wide)
        correct_chromas_strict = count_nonzero_fn(correct_chromas_strict)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'wide', correct_chromas_wide, viterbi=True)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', 'strict', correct_chromas_strict, viterbi=True)

        assert all(self.viterbi_var_dict['all_updated'].values())

        est_notes_with_voicing_info = tf.where(est_voicing, est_notes, -est_notes)

        return est_notes_with_voicing_info

    def update_states(self, rec_idx, snippet_idx, ref_notes, logits, loss):

        model = self.model

        t = [rec_idx, snippet_idx, ref_notes, logits, loss]
        assert all(isinstance(v, tf.Tensor) for v in t)

        _rec_idx = rec_idx.numpy()
        rec_dict = model.tf_dataset.np_dataset[_rec_idx]
        num_snippets = len(rec_dict['split_list'])
        snippet_idx.set_shape([])
        _snippet_idx = snippet_idx.numpy()
        assert _snippet_idx < num_snippets

        if _snippet_idx == 0:
            self.rec_idx = _rec_idx
            self.snippet_idx = _snippet_idx
            self.est_notes_with_voicing_info = []

            self.logits_list = []
            self.ref_notes_list = []

        assert _rec_idx == self.rec_idx

        if _snippet_idx > 0:
            assert _snippet_idx == self.snippet_idx + 1

        self.snippet_idx = _snippet_idx

        est_notes_with_voicing_info = self.update_states_tf_fn(
            rec_idx=rec_idx,
            ref_notes=ref_notes,
            logits=logits,
            loss=loss
        )

        self.est_notes_with_voicing_info.append(est_notes_with_voicing_info.numpy())


        self.logits_list.append(logits.numpy())
        self.ref_notes_list.append(ref_notes.numpy())

        if _snippet_idx == num_snippets - 1:
            est_notes_with_voicing_info = np.concatenate(self.est_notes_with_voicing_info)
            num_frames = len(rec_dict['spectrogram'])
            assert est_notes_with_voicing_info.shape == (num_frames,)
            oa = self.mir_eval_oa_fn(self.rec_idx, est_notes_with_voicing_info)
            self.mir_eval_oas.append(oa)

            no_viterbi_est_voicing = est_notes_with_voicing_info > 0
            no_viterbi_est_notes = est_notes_with_voicing_info

        if _snippet_idx == num_snippets - 1:
            logits = np.concatenate(self.logits_list)
            ref_notes = np.concatenate(self.ref_notes_list)
            assert len(logits) == len(ref_notes)
            n_frames = self.model.tf_dataset.num_frames_vector[self.rec_idx]
            assert len(logits) == n_frames
            voiced_flags, bins = MetricsInference.viterbi(logits)
            est_notes_with_voicing_info = self.viterbi_update_states_tf_fn(self.rec_idx, ref_notes, logits, melody_bins=bins, est_voicing=voiced_flags)
            est_notes_with_voicing_info = est_notes_with_voicing_info.numpy()
            oa = self.mir_eval_oa_fn(self.rec_idx, est_notes_with_voicing_info)
            self.viterbi_mir_eval_oas.append(oa)

            viterbi_est_voicing = voiced_flags
            viterbi_est_notes = bins

        # if _snippet_idx == num_snippets - 1:
        #     self.effect_of_viterbi_fn(
        #         viterbi_est_voicing=viterbi_est_voicing,
        #         viterbi_est_notes=viterbi_est_notes,
        #         no_viterbi_est_voicing=no_viterbi_est_voicing,
        #         no_viterbi_est_notes=no_viterbi_est_notes
        #     )

    def results(self):

        model = self.model
        num_recs = self.num_recs
        melody_dict = self.var_dict['melody']
        var_loss = self.var_dict['loss']
        var_batch_counter = self.var_dict['batch_counter']
        f8f4div = self.to_f8_divide_and_to_f4_fn
        num_frames_vector = tf.convert_to_tensor(model.tf_dataset.num_frames_vector, tf.int64)

        correct_frames = melody_dict['correct_pitches']['strict'] + melody_dict['voicing']['correct_unvoiced']
        _num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        tf.debugging.assert_equal(_num_frames_vector, num_frames_vector)
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
        m_loss = var_loss / tf.cast(var_batch_counter, tf.float32)

        self.tf_oas = m_oa.numpy()
        self.oa = tf.reduce_mean(m_oa).numpy().item()
        self.loss = m_loss.numpy().item()
        self.current_voicing_threshold = self.voicing_threshold.read_value().numpy().item()

        results = dict(
            loss=m_loss,
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

    def viterbi_results(self):

        model = self.model
        num_recs = self.num_recs
        melody_dict = self.viterbi_var_dict['melody']
        f8f4div = self.to_f8_divide_and_to_f4_fn
        num_frames_vector = tf.convert_to_tensor(model.tf_dataset.num_frames_vector, tf.int64)

        correct_frames = melody_dict['correct_pitches']['strict'] + melody_dict['voicing']['correct_unvoiced']
        _num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        tf.debugging.assert_equal(_num_frames_vector, num_frames_vector)
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

        self.viterbi_tf_oas = m_oa.numpy()

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

    @staticmethod
    def est_notes_with_voicing_info_to_hz_fn(est_notes):

        min_note = TFDataset.note_min

        larger = est_notes >= min_note
        smaller = est_notes <= -min_note
        larger_or_smaller = np.logical_or(larger, smaller)
        assert np.all(larger_or_smaller)

        positives = np.where(est_notes >= min_note)
        negatives = np.where(est_notes <= -min_note)

        freqs = np.empty_like(est_notes)
        freqs[positives] = librosa.midi_to_hz(est_notes[positives])
        freqs[negatives] = -librosa.midi_to_hz(-est_notes[negatives])

        return freqs

    def mir_eval_oa_fn(self, rec_idx, est_notes_with_voicing_info):

        model = self.model

        rec_dict = model.tf_dataset.np_dataset[rec_idx]
        ref_times = rec_dict['original']['times']
        ref_freqs = rec_dict['original']['freqs']

        est_notes = est_notes_with_voicing_info
        num_frames = len(est_notes)
        est_times = np.arange(num_frames) * (256. / 44100.)
        est_freqs = MetricsInference.est_notes_with_voicing_info_to_hz_fn(est_notes)

        oa = mir_eval.melody.evaluate(
            ref_time=ref_times,
            ref_freq=ref_freqs,
            est_time=est_times,
            est_freq=est_freqs
        )['Overall Accuracy']

        return oa

    def effect_of_viterbi_fn(self, viterbi_est_voicing, viterbi_est_notes, no_viterbi_est_voicing, no_viterbi_est_notes):

        rec_idx = self.rec_idx

        ref_notes = self.model.tf_dataset.np_dataset[rec_idx]['notes']

        assert len(viterbi_est_voicing) == len(viterbi_est_notes) == len(no_viterbi_est_voicing) == len(no_viterbi_est_notes)

        ref_notes = ref_notes.copy()
        ref_notes[ref_notes == 0] = np.nan
        viterbi_est_notes = viterbi_est_notes / 5. + TFDataset.note_min
        viterbi_est_notes[np.logical_not(viterbi_est_voicing)] = np.nan
        no_viterbi_est_notes[np.logical_not(no_viterbi_est_voicing)] = np.nan

        fig, (ax_ref, ax_viterbi, ax_no) = plt.subplots(3, sharex=True)
        x = list(range(len(ref_notes)))
        for ax, name, y in zip(
                (ax_ref, ax_viterbi, ax_no),
                ('reference', 'viterbi', 'w/o viterbi'),
                (ref_notes, viterbi_est_notes, no_viterbi_est_notes)):
            ax.scatter(x=x, y=y, s=.5, c='k')
            ax.set_ylabel(name)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel('time')

        rec_name = self.model.tf_dataset.rec_names[rec_idx]
        fig.suptitle(rec_name)
        plt.savefig(rec_name + '.png')
        plt.close()


class TBSummary:

    def __init__(self, model):

        assert hasattr(model, 'metrics')

        self.model = model
        self.is_inferencing = model.config.train_or_inference.inference is not None

        self.tb_path = os.path.join(model.config.tb_dir, model.name)
        self.tb_summary_writer = tf.summary.create_file_writer(self.tb_path)

        if hasattr(model.tf_dataset, 'rec_names'):
            self.rec_names = model.tf_dataset.rec_names
            self.num_recs = len(self.rec_names)

        self.header = ['vrr', 'vfa', 'va', 'rpa_strict', 'rpa_wide', 'rca_strict', 'rca_wide', 'oa']
        self.num_columns = len(self.header)

        self.table_ins = self.create_tf_table_writer_ins_fn()

        if self.is_inferencing:
            self.viterbi_table_ins = self.create_tf_table_writer_ins_fn()

    def create_tf_table_writer_ins_fn(self):

        is_inferencing = self.is_inferencing
        model = self.model
        header = self.header
        description = 'metrics'
        tb_summary_writer = self.tb_summary_writer

        if hasattr(self, 'rec_names'):
            assert is_inferencing or not is_inferencing and not model.is_training
            names = list(self.rec_names) + ['average']
            table_ins = ArrayToTableTFFn(
                writer=tb_summary_writer,
                header=header,
                scope=description,
                title=description,
                names=names
            )
        else:
            assert not is_inferencing and model.is_training
            table_ins = ArrayToTableTFFn(
                writer=tb_summary_writer,
                header=header,
                scope=description,
                title=description,
                names=['average']
            )

        return table_ins

    def prepare_table_data_fn(self, result_dict):

        header = self.header

        if hasattr(self, 'rec_names'):

            data = [result_dict[name] for name in header]
            data = tf.stack(data, axis=-1)
            tf.ensure_shape(data, [self.num_recs, self.num_columns])
            ave = tf.reduce_mean(data, axis=0, keepdims=True)
            data = tf.concat([data, ave], axis=0)

        else:
            data = [result_dict[name] for name in header]
            data = [data]
            data = tf.convert_to_tensor(data)

        return data

    def viterbi_prepare_table_data_fn(self, result_dict):

        header = self.header

        assert hasattr(self, 'rec_names')
        data = [result_dict[name] for name in header]
        data = tf.stack(data, axis=-1)
        tf.ensure_shape(data, [self.num_recs, self.num_columns])
        ave = tf.reduce_mean(data, axis=0, keepdims=True)
        data = tf.concat([data, ave], axis=0)

        return data

    def write_tb_summary_fn(self, step_int):

        model = self.model
        is_inferencing = self.is_inferencing

        assert isinstance(step_int, int)

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                result_dict = model.metrics.results()

                if not is_inferencing:
                    with self.tb_summary_writer.as_default():
                        for metric_name in ('loss', 'oa'):
                            value = getattr(model.metrics, metric_name)
                            assert value is not None
                            tf.summary.scalar(metric_name, value, step=step_int)
                        if not model.is_training:
                            value = model.metrics.current_voicing_threshold
                            assert value is not None
                            tf.summary.scalar('voicing_threshold', value, step=step_int)

                else:
                    with self.tb_summary_writer.as_default():
                        loss = model.metrics.loss
                        assert loss is not None
                        tf.summary.text('loss', str(loss), step=step_int)

                data = self.prepare_table_data_fn(result_dict)
                self.table_ins.write(data, step_int)


                if self.is_inferencing:
                    with tf.name_scope('viterbi'):
                        viterbi_result_dict = model.metrics.viterbi_results()
                        data = self.viterbi_prepare_table_data_fn(viterbi_result_dict)
                        self.viterbi_table_ins.write(data, step_int)


class Model:

    def __init__(self, config, name):

        assert name in config.model_names

        inferencing = config.train_or_inference.inference is not None

        if not inferencing:
            assert 'test' not in name

        self.name = name
        self.is_training = True if 'train' in name else False
        self.config = config

        if inferencing:
            if self.name != 'test':
                self.tf_dataset = TFDatasetForInferenceMode(self)
            else:
                self.tf_dataset = TFDatasetForRWC(self)
        else:
            if self.is_training:
                self.tf_dataset = TFDatasetForTrainingModeTrainingSplit(self)
            else:
                self.tf_dataset = TFDatasetForInferenceMode(self)

        if inferencing:
            self.metrics = MetricsInference(self)
        else:
            if self.is_training:
                self.metrics = MetricsTrainingModeTrainingSplit(self)
            else:
                self.metrics = MetricsValidation(self)

        self.tb_summary_ins = TBSummary(self)


def main():

    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    for name in MODEL_DICT['config'].model_names:
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    aug_info = []
    aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
    aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug_mode))
    aug_info.append('snippet length - {}'.format(MODEL_DICT['config'].snippet_len))
    if MODEL_DICT['config'].train_or_inference.inference is None:
        aug_info.append('batch size - 1')
        aug_info.append('num of batches per epoch - {}'.format(MODEL_DICT['config'].batches_per_epoch))
        aug_info.append('num of patience epochs - {}'.format(MODEL_DICT['config'].patience_epochs))
        aug_info.append('initial learning rate - {}'.format(MODEL_DICT['config'].initial_learning_rate))
    aug_info = '\n\n'.join(aug_info)
    logging.info(aug_info)
    with MODEL_DICT['training'].tb_summary_ins.tb_summary_writer.as_default():
        tf.summary.text('auxiliary_information', aug_info, step=0)

    def training_fn(global_step=None):

        assert isinstance(global_step, int)

        config = MODEL_DICT['config']
        model = MODEL_DICT['training']

        assert config.train_or_inference.inference is None
        assert model.is_training

        iterator = model.tf_dataset.iterator
        acoustic_model = config.acoustic_model_ins
        trainable_vars = acoustic_model.trainable_variables
        metrics = model.metrics
        write_tb_summary_fn = model.tb_summary_ins.write_tb_summary_fn
        batches_per_epoch = config.batches_per_epoch
        optimizer = model.config.optimizer
        loss_fn = acoustic_model.loss_tf_fn

        metrics.reset()
        for batch_idx in range(batches_per_epoch):
            logging.debug('batch {}/{}'.format(batch_idx + 1, batches_per_epoch))
            batch = iterator.get_next()
            with tf.GradientTape() as tape:
                logits = acoustic_model(batch['spectrogram'], training=True)
                loss = loss_fn(ref_notes=batch['notes'][0], logits=logits)
            grads = tape.gradient(loss, trainable_vars)
            grads = acoustic_model.add_wd_grad_fn(grads)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            metrics.update_states(ref_notes=batch['notes'][0], logits=logits, loss=loss)
        write_tb_summary_fn(global_step)

        loss = model.metrics.loss
        oa = model.metrics.oa
        v_th = model.metrics.current_voicing_threshold
        logging.info('{} - step - {} - loss - {} - oa - {} - voicing threshold - {}'.format(model.name, global_step, loss, oa, v_th))

    def inference_fn(model_name, global_step=None):

        config = MODEL_DICT['config']

        is_inferencing = config.train_or_inference.inference is not None

        if not is_inferencing:
            assert model_name == 'validation'

        assert isinstance(global_step, int)

        model = MODEL_DICT[model_name]

        acoustic_model = config.acoustic_model_ins
        assert not hasattr(model.tf_dataset, 'iterator')
        iterator = iter(model.tf_dataset.tf_dataset)
        metrics = model.metrics
        batches_per_epoch = len(model.tf_dataset.rec_start_end_idx_list)
        loss_fn = acoustic_model.loss_tf_fn

        metrics.reset()
        for batch_idx in range(batches_per_epoch):

            batch = iterator.get_next()
            logits = acoustic_model(batch['spectrogram'], training=False)
            loss = loss_fn(ref_notes=batch['notes'][0], logits=logits)
            metrics.update_states(
                rec_idx=batch['rec_idx'][0],
                snippet_idx=batch['snippet_idx'][0],
                ref_notes=batch['notes'][0],
                logits=logits,
                loss=loss
            )
        batch = iterator.get_next_as_optional()
        assert not batch.has_value()

        model.tb_summary_ins.write_tb_summary_fn(global_step)

        loss = model.metrics.loss
        oa = model.metrics.oa
        v_th = model.metrics.current_voicing_threshold
        logging.info('{} - step - {} - loss - {} - oa - {} - voicing threshold - {}'.format(model.name, global_step, loss, oa, v_th))

        if is_inferencing:
            mir_eval_oas = metrics.mir_eval_oas
            mir_eval_oas = np.asarray(mir_eval_oas)
            tf_oas = metrics.tf_oas

            oa_diffs = tf_oas - mir_eval_oas

            print('tf and mir_eval oas and their differences -')
            for idx in range(len(tf_oas)):
                print(idx, tf_oas[idx], mir_eval_oas[idx], oa_diffs[idx])
            tf_oa = np.mean(tf_oas)
            mir_eval_oa = np.mean(mir_eval_oas)
            diff = tf_oa - mir_eval_oa
            print('ave', tf_oa, mir_eval_oa, diff)

            print()
            print('viterbi tf and mir_eval oas and their difference -')
            viterbi_tf_oas = metrics.viterbi_tf_oas
            viterbi_mir_eval_oas = metrics.viterbi_mir_eval_oas
            viterbi_oa_differences = viterbi_tf_oas - viterbi_mir_eval_oas
            for idx in range(len(viterbi_tf_oas)):
                print(idx, viterbi_tf_oas[idx], viterbi_mir_eval_oas[idx], viterbi_oa_differences[idx])
            viterbi_tf_ave_oa = np.mean(viterbi_tf_oas)
            viterbi_mir_eval_ave_oa = np.mean(viterbi_mir_eval_oas)
            diff = viterbi_tf_ave_oa - viterbi_mir_eval_ave_oa
            print('ave', viterbi_tf_ave_oa, viterbi_mir_eval_ave_oa, diff)

    if MODEL_DICT['config'].train_or_inference.inference is not None:
        ckpt_file = MODEL_DICT['config'].train_or_inference.inference
        ckpt_dir, ckpt_name = os.path.split(ckpt_file)
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
        ckpt = tf.train.Checkpoint(model=MODEL_DICT['config'].acoustic_model_ins.model_for_ckpt)
        status = ckpt.restore(ckpt_file)
        status.expect_partial()
        status.assert_existing_objects_matched()

        logging.info('inferencing ... ')
        for model_name in MODEL_DICT['config'].model_names:
            logging.info(model_name)
            inference_fn(model_name, global_step=0)
            MODEL_DICT[model_name].tb_summary_ins.tb_summary_writer.close()

    elif MODEL_DICT['config'].train_or_inference.from_ckpt is not None:
        ckpt = tf.train.Checkpoint(
            model=MODEL_DICT['config'].acoustic_model_ins.model_for_ckpt,
            optimizer=MODEL_DICT['config'].optimizer
        )
        ckpt_file = MODEL_DICT['config'].train_or_inference.from_ckpt
        ckpt_dir, ckpt_name = os.path.split(ckpt_file)
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
        status = ckpt.restore(ckpt_file)
        assert status.assert_consumed()
        logging.info('reproducing results ...')

        model_name = 'validation'
        logging.info(model_name)
        inference_fn(model_name, global_step=0)
        best_oa = MODEL_DICT[model_name].metrics.oa
        assert best_oa is not None
        best_epoch = 0
    else:
        logging.info('training from scratch ...')
        best_oa = None

    # training
    if MODEL_DICT['config'].train_or_inference.inference is None:

        assert MODEL_DICT['config'].train_or_inference.ckpt_prefix is not None
        assert 'ckpt_manager' not in MODEL_DICT
        ckpt = tf.train.Checkpoint(
            model=MODEL_DICT['config'].acoustic_model_ins.model_for_ckpt,
            optimizer=MODEL_DICT['config'].optimizer
        )
        ckpt_dir, ckpt_prefix = os.path.split(MODEL_DICT['config'].train_or_inference.ckpt_prefix)
        assert ckpt_prefix != ''
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
        ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            directory=ckpt_dir,
            max_to_keep=1,
            checkpoint_name=ckpt_prefix
        )
        MODEL_DICT['ckpt_manager'] = ckpt_manager

        patience_epochs = MODEL_DICT['config'].patience_epochs
        training_epoch = 1

        while True:

            logging.info('\nepoch - {}'.format(training_epoch))

            for model_name in MODEL_DICT['config'].model_names:

                logging.info(model_name)

                if 'train' in model_name:
                    training_fn(training_epoch)
                elif 'validation' in model_name:
                    inference_fn(model_name, training_epoch)

            valid_oa = MODEL_DICT['validation'].metrics.oa
            should_save = best_oa is None or best_oa < valid_oa
            if should_save:
                best_oa = valid_oa
                best_epoch = training_epoch
                save_path = MODEL_DICT['ckpt_manager'].save(checkpoint_number=training_epoch)
                logging.info('weights checkpointed to {}'.format(save_path))

            d = training_epoch - best_epoch
            if d >= patience_epochs:
                logging.info('training terminated at epoch {}'.format(training_epoch))
                break

            training_epoch = training_epoch + 1

        for model_name in MODEL_DICT['config'].model_names:
            model = MODEL_DICT[model_name]
            model.tb_summary_ins.tb_summary_writer.close()


if __name__ == '__main__':

    main()

