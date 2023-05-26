
"""
Hsieh model, vocal melody, only mdb


old for IEEE SPL:
adc04: 0.8423164188718233
mirex05: 0.7953167491048908
mdb: 0.8442/0.7719/0.7910  loss: 1.1228034/1.432828/1.4580796
mir1k: 0.6060539620900695
rwc: 0.725741772709556

singer:
adc04: 0.8356
mirex05: 0.7716
mdb: 0.8007/0.7567/0.7777 loss: 1.3409446/1.5099791/1.5681676
mir1k: 0.5921
rwc: 0.7004


m2m3:
adc04: 0.8431
mirex05: 0.7819
mdb: 0.7977/0.7782/0.7629 loss: 1.360434/1.4886932/1.7062963
mir1k: 0.5945
rwc: 0.7046




31-Mar-2023:
before with automatic voicing detection:
training/validation/test oas: 0.7977/0.7782/0.7629

now with deterministic voicing threshold:
no interpolation on est notes
voicing threshold 0.54
validation oa 0.7805
training/validation/test oas: 0.7994/0.7805/0.7663

applying viterbi:
test on more datasets:
dataset: w/o w/
mdb: 0.7663 0.7775
adb04: 0.8415 0.8435
mirex05: 0.7816 0.7803
mir1k: 0.5969 0.6026
rwc: 0.7038 0.7022


21-May-2023 Sun
mir_eval

softmax-scaled
training/validation/test  0.5313/0.4940/0.4825

softmax-unscaled
0.7753/0.7665/0.7584


"""

DEBUG = False

import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import tensorflow as tf
tf.random.set_seed(20220909)
from argparse import Namespace
import glob
import os
import numpy as np
import librosa
from self_defined import ArrayToTableTFFn
import acoustic_model_correct as acoustic_model_module
import soundfile
import mir_eval
from tf_cfp import CFP as CFPClass
import medleydb as mdb
from self_defined import is_vocals_m2m3_fn as is_vocals_fn
from self_defined import load_np_array_from_file_fn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


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
        Config.set_gpu_fn()

        self.allowed_viterbi_methods = ('shaun', 'softmax-scaled', 'softmax-unscaled')
        self.viterbi_method = self.allowed_viterbi_methods[2]

        self.train_or_inference = Namespace(
            inference='/media/ssd/music_trans/2209/9_hsieh/m2m3/ckpts/d0-13',
            from_ckpt=None,
            ckpt_prefix=None
        )
        self.tb_dir = 'tb_inf_rwc'

        if self.train_or_inference.inference is None:
            self.model_names = ('training', 'validation')
        else:
            self.model_names = ('training', 'validation', 'test')

        # check if tb_dir and checkpoints with the same prefix already exist
        if not self.debug_mode:
            self.chk_if_tb_dir_and_model_with_same_prefix_exist_fn()

        self.snippet_len = 10 * 430
        self.initial_learning_rate = 1e-4
        self.batches_per_epoch = None if self.debug_mode else 250
        self.patience_epochs = 20

        self.tvt_split_dict = Config.get_mdb_split_fn()
        if self.debug_mode:
            for k in self.tvt_split_dict:
                self.tvt_split_dict[k] = self.tvt_split_dict[k][:1]

        self.acoustic_model_ins = AcousticModel(self)

        note_range = TFDataset.note_range
        # build optimizer so as to create relevant optimizer weights
        if self.train_or_inference.inference is None:
            self.learning_rate = tf.Variable(
                self.initial_learning_rate, dtype=tf.float32, name='learning_rate', trainable=False)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            spec = tf.random.normal([1, self.snippet_len, 320, 3])
            lower_note = note_range[0]
            upper_note = note_range[-1]
            labels = np.random.uniform(lower_note, upper_note, [self.snippet_len])
            with tf.GradientTape() as tape:
                logits = self.acoustic_model_ins(spec, training=False)
                loss = self.acoustic_model_ins.loss_tf_fn(ref_notes=labels, logits=logits)
            grads = tape.gradient(loss, self.acoustic_model_ins.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.acoustic_model_ins.trainable_variables))
            assert len(self.optimizer.weights) > 0

            if self.debug_mode:
                print('weights of the optimizer: ')
                for idx, w in enumerate(self.optimizer.weights):
                    print(idx, w.name, w.shape)
                print()

    @staticmethod
    def set_gpu_fn():

        gpus = tf.config.list_physical_devices('GPU')
        num_gpus = len(gpus)
        assert num_gpus == 1
        tf.config.experimental.set_memory_growth(gpus[0], True)

    def chk_if_tb_dir_and_model_with_same_prefix_exist_fn(self):

        # check if tb_dir exists
        assert self.tb_dir is not None
        is_tb_dir_exist = glob.glob('{}/'.format(self.tb_dir))
        if is_tb_dir_exist:
            assert False, 'directory {} already exists'.format(self.tb_dir)

        # check if model exists
        if self.train_or_inference.inference is None and self.train_or_inference.ckpt_prefix is not None:
            ckpt_dir, ckpt_prefix = os.path.split(self.train_or_inference.ckpt_prefix)
            assert ckpt_prefix != ''
            if ckpt_dir == '':
                ckpt_dir = 'ckpts'

            is_exist = glob.glob('{}/{}*'.format(ckpt_dir, ckpt_prefix))
            if is_exist:
                assert False, 'checkpoints with prefix {} already exist'.format(ckpt_prefix)

    @staticmethod
    def gen_central_notes_fn():

        fmin = 31
        fmax = 1250
        bins_per_oct = 60
        central_freqs = []

        fac = 2. ** (1. / bins_per_oct)
        f = float(fmin)
        while f < fmax:
            central_freqs.append(f)
            f = f * fac
        central_freqs = central_freqs[1:]
        central_freqs = librosa.hz_to_midi(central_freqs)
        assert len(central_freqs) == 320
        central_freqs = central_freqs.astype(np.float32)
        central_freqs.flags['WRITEABLE'] = False

        return central_freqs

    @staticmethod
    def get_mdb_split_fn():

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

        test_songlist = ["daisy1", "daisy2", "daisy3", "daisy4", "opera_fem2", "opera_fem4", "opera_male3",
                         "opera_male5", "pop1", "pop2", "pop3", "pop4"]
        assert len(test_songlist) == 12

        return test_songlist

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


class AcousticModel:

    def __init__(self, config):

        acoustic_model = acoustic_model_module.create_acoustic_model_fn()

        if config.debug_mode:
            acoustic_model.summary(line_length=150)
            print('trainable variables - ')
            for idx, w in enumerate(acoustic_model.trainable_variables):
                print(idx, w.name, w.shape)

        assert len(acoustic_model.trainable_variables) > 0

        self.acoustic_model = acoustic_model
        self.trainable_variables = acoustic_model.trainable_variables

        self.voicing_threshold = 0.54

        self.model_for_ckpt = dict(acoustic_model=acoustic_model)
        self.config = config
        self.cutoff_prob = 4e-3  # p[0, .2, .4, .6] = [1., .54, .085, .0039]

        viterbi_method = config.viterbi_method
        if viterbi_method == 'shaun':
            self.viterbi = Viterbi(self.voicing_threshold)
        elif viterbi_method == 'softmax-scaled':
            self.viterbi = SoftMaxViterbi(scaled=True)
        elif viterbi_method == 'softmax-unscaled':
            self.viterbi = SoftMaxViterbi(scaled=False)
        else:
            assert False

    def __call__(self, inputs, training):

        assert isinstance(training, bool)
        outputs = inputs

        outputs = self.acoustic_model(outputs, training=training)

        return outputs

    @staticmethod
    def ref_notes_larger_than_min_note_check_fn(ref_notes):

        ref_notes.set_shape([None])
        assert ref_notes.dtype == tf.float32

        t1 = ref_notes == 0.
        t2 = ref_notes >= TFDataset.note_range[0]
        t1 = tf.logical_or(t1, t2)
        tf.debugging.assert_equal(t1, True)

    @staticmethod
    def ref_most_suitable_idx_to_label_fn(most_suitable_indices):

        most_suitable_indices.set_shape([None])
        assert most_suitable_indices.dtype == tf.int32

        num_frames = tf.size(most_suitable_indices, out_type=tf.int32)
        t = tf.range(num_frames, dtype=tf.int32)
        most_suitable_indices = tf.stack([t, most_suitable_indices], axis=-1)
        most_suitable_indices.set_shape([None, 2])

        labels = tf.scatter_nd(shape=[num_frames, 321], indices=most_suitable_indices, updates=tf.ones([num_frames]))

        return labels

    @tf.function(input_signature=[
        tf.TensorSpec([None]),
        tf.TensorSpec([None, 321])
    ], autograph=False)
    def loss_tf_fn(self, ref_notes, logits):
        # smoothing

        note_range = TFDataset.note_range
        assert len(note_range) == 320 and note_range[0] > 0

        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        logits = tf.convert_to_tensor(logits, tf.float32)

        max_note = note_range[-1] + .4
        max_note = max_note.astype(np.float32)
        ref_notes = tf.minimum(ref_notes, max_note)

        AcousticModel.ref_notes_larger_than_min_note_check_fn(ref_notes)

        note_range = np.pad(note_range, [[1, 0]])
        note_range = tf.convert_to_tensor(note_range, tf.float32)
        ref_notes = ref_notes[:, None] - note_range[None, :]
        ref_notes.set_shape([None, 321])
        t = - ref_notes ** 2 / (2. * .18 ** 2)
        ref_notes = tf.exp(t)
        ref_notes = tf.where(ref_notes < self.cutoff_prob, tf.zeros_like(ref_notes), ref_notes)
        t = tf.reduce_sum(ref_notes, axis=1, keepdims=True)
        tf.debugging.assert_greater(t, .08)
        ref_notes = ref_notes / t

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=ref_notes, logits=logits, axis=1)
        loss.set_shape([None])
        loss = tf.reduce_mean(loss)

        return loss


class TFDataset:

    medleydb_dir = os.environ['medleydb']
    melody2_dir = os.environ['melody2_dir']
    mix_suffix = '_MIX.wav'
    melody2_suffix = '_MELODY2.csv'
    note_range = Config.gen_central_notes_fn()
    assert len(note_range) == 320 and note_range[0] > 0
    note_min = note_range[0]
    freq_min = librosa.midi_to_hz(note_min)

    cfp_fn = CFPClass()

    def __init__(self, model):

        self.model = model

    @staticmethod
    def gen_spec_fn(track_id):

        wav_file = os.path.join(TFDataset.medleydb_dir, track_id, track_id + TFDataset.mix_suffix)
        spec = TFDataset.cfp_fn(wav_file)

        return spec

    @staticmethod
    def hz_to_midi_fn(freqs):

        assert np.all(freqs >= 0)
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

        is_vocals = is_vocals_fn(track_id)
        assert len(is_vocals) == len(m2_melody)

        m2_freqs = m2_melody[:, 1]
        TFDataset.validity_check_of_ref_freqs_fn(m2_freqs)

        vocal_freqs = np.where(is_vocals, m2_freqs, 0.)
        notes = TFDataset.hz_to_midi_fn(vocal_freqs)

        result = dict(notes=notes, original=dict(times=m2_melody[:, 0], freqs=vocal_freqs))

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
            logging.info('{}/{}'.format(track_idx + 1, num_tracks))

            spec = TFDataset.gen_spec_fn(track_id)
            notes_original_dict = TFDataset.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - len(spec)
            assert 0 <= diff <= 1
            if diff == 1:
                spec = np.pad(spec, [[0, 1], [0, 0], [0, 0]])

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        return dataset

    def note_out_of_range_chk_fn(self, np_dataset):

        model = self.model
        note_range = TFDataset.note_range
        assert len(note_range) == 320 and note_range[0] > 0

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

        min_melody_freq = TFDataset.freq_min

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
        spec.set_shape([None, 320, 3])
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
        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)
        num_frames = [len(rec_dict['spectrogram']) for rec_dict in self.np_dataset]
        self.num_frames_vector = np.asarray(num_frames, dtype=np.int64)
        self.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_split_list_fn(self.np_dataset)
        self.batches_per_epoch = len(self.rec_start_end_idx_list)
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
        spec.set_shape([None, 320, 3])
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
        self.batches_per_epoch = len(self.rec_start_end_idx_list)
        self.tf_dataset = self.gen_tf_dataset_fn()

    @staticmethod
    def gen_spec_fn(track_id):

        mix_wav_file = os.path.join(os.environ['adc04'], track_id + '.wav')

        return TFDataset.cfp_fn(mix_wav_file)

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
        spec.set_shape([None, 320, 3])
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
        self.batches_per_epoch = len(self.rec_start_end_idx_list)
        self.tf_dataset = self.gen_tf_dataset_fn()

    @staticmethod
    def gen_spec_fn(track_id):

        mix_wav_file = os.path.join(os.environ['mirex05'], track_id + '.wav')

        return TFDataset.cfp_fn(mix_wav_file)

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
                spec = np.pad(spec, [[0, diff], [0, 0], [0, 0]])
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
        spec.set_shape([None, 320, 3])
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
        self.batches_per_epoch = len(self.rec_start_end_idx_list)
        self.tf_dataset = self.gen_tf_dataset_fn()

    @staticmethod
    def gen_spec_fn(track_id):

        wav_file = os.path.join(os.environ['mir1k'], 'Wavfile', track_id + '.wav')
        spec = TFDataset.cfp_fn(wav_file)

        return spec

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
        spec.set_shape([None, 320, 3])
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
        self.batches_per_epoch = len(self.rec_start_end_idx_list)
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
        assert rec_idx >= 0

        aiff_file = self.rec_files[rec_idx]

        return TFDataset.cfp_fn(aiff_file)

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
                spec = np.pad(spec, [[0, 1], [0, 0], [0, 0]])
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
        spec.set_shape([None, 320, 3])
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
    def est_notes_interpolation_fn(est_peak_indices, est_probs=None, interpolation=False):

        if interpolation:

            note_range = TFDataset.note_range
            note_offset = note_range[0]
            assert note_offset > 23
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
        else:
            note_range = TFDataset.note_range
            assert note_range[0] > 23
            note_range = tf.convert_to_tensor(note_range, tf.float32)
            est_notes = tf.gather(note_range, est_peak_indices)

        return est_notes

    @staticmethod
    def octave(distance):

        distance = tf.floor(distance / 12. + .5) * 12.

        return distance


class Viterbi:

    def __init__(self, voicing_threshold):

        self.num_freq_bins = 320
        self.h = 256
        self.sr = 44100
        self.single_side_peak_width = 5

        th = voicing_threshold
        assert th > 0
        assert th < 1
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


class SoftMaxViterbi:

    def __init__(self, scaled):

        """
        :param voicing_threshold: a probability threshold
        """

        self.scaled = scaled

        self.num_freq_bins = 320
        self.single_side_peak_width = 5

        self.log_transition_matrix_T = self.transition_matrix_fn()
        self.ini_probs, self.log_ini_probs = self.init_probs_fn()

    def find_peaks_all_at_once_np_fn(self, logits):

        n_bins = self.num_freq_bins
        spw = self.single_side_peak_width

        frames_logits = logits
        assert frames_logits.ndim == 2
        assert frames_logits.shape[1] == n_bins + 1
        frames_logits = frames_logits[:, 1:]
        n_frames = len(frames_logits)
        frames_logits = np.pad(frames_logits, [(0, 0), (spw, spw)], mode='reflect')
        w = 2 * spw + 1
        frames_are_peaks = np.zeros([n_frames, n_bins + 1], np.bool_)
        frames_are_peaks[:, 0] = True
        spw = np.asarray(spw, np.int64)
        for bin_idx in range(n_bins):
            peak_idx = np.argmax(frames_logits[:, bin_idx:bin_idx + w], axis=1)
            are_peaks = peak_idx == spw
            frames_are_peaks[:, bin_idx + 1] = are_peaks

        return frames_are_peaks

    def observation_probs_fn(self, logits):

        n_bins = self.num_freq_bins

        if self.scaled:
            ini_probs = self.ini_probs
            assert ini_probs.min() > 0.3 / (n_bins * 10)
            ini_probs = np.roll(ini_probs, 1)
            ini_probs = ini_probs.astype(np.float32)
        else:
            ini_probs = np.ones([n_bins + 1], np.float32)

        assert isinstance(logits, np.ndarray)
        assert logits.dtype == np.float32
        assert logits.ndim == 2
        assert logits.shape[1] == n_bins + 1
        assert logits.flags['C_CONTIGUOUS']

        n_frames = len(logits)

        prob_ts = np.zeros([n_frames, 1 + n_bins], np.float32)
        are_peaks_ts = self.find_peaks_all_at_once_np_fn(logits)

        for frame_idx, are_peaks in enumerate(are_peaks_ts):
            peak_indices = np.where(are_peaks)[0]
            n_peaks = len(peak_indices)
            if n_peaks == 1:
                assert peak_indices[0] == 0
                prob_ts[frame_idx, 0] = 1. / ini_probs[0]
                continue

            peak_logits = logits[frame_idx, peak_indices]
            max_logit = np.max(peak_logits)
            np.subtract(peak_logits, max_logit, out=peak_logits)
            np.exp(peak_logits, out=peak_logits)
            t= np.sum(peak_logits)
            np.divide(peak_logits, t, out=peak_logits)
            priors = ini_probs[peak_indices]
            np.divide(peak_logits, priors, out=peak_logits)
            prob_ts[frame_idx, peak_indices] = peak_logits

        # t = np.sum(prob_ts, axis=1)
        # assert np.allclose(t, 1)
        prob_ts = np.roll(prob_ts, shift=-1, axis=1)

        return prob_ts

    def init_probs_fn(self):

        U = self.num_freq_bins

        file_name = 'viterbi_init_probs.dat'
        _name, probs = load_np_array_from_file_fn(file_name)
        assert _name == file_name[:-4]
        assert probs.shape == (U + 1,)
        assert np.argmax(probs) == U
        assert np.isclose(np.sum(probs), 1)
        assert np.all(probs > 0)

        tiny = np.finfo(np.float32).tiny
        t = np.log(probs + tiny)
        assert not np.any(np.isneginf(t))
        t = np.require(t, np.float32)
        t.flags['WRITEABLE'] = False

        return probs, t

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

        prob_ts = self.observation_probs_fn(logits)
        bins = self.viterbi_librosa_fn(prob_ts)

        voiced = bins < n_bins
        bins = np.minimum(bins, n_bins - 1)

        return voiced, bins

    def viterbi_librosa_fn(self, probs_ts):

        """
        np version is faster than tf version

        """
        S = self.num_freq_bins + 1
        B = self.log_transition_matrix_T
        prob_init = self.log_ini_probs

        assert probs_ts.ndim == 2
        assert probs_ts.shape[1] == S
        assert probs_ts.dtype == np.float32
        assert probs_ts.flags['C_CONTIGUOUS']
        tinyp = np.finfo(np.float32).tiny
        np.add(probs_ts, tinyp, out=probs_ts)
        np.log(probs_ts, out=probs_ts)
        probs = probs_ts
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


class MetricsInference(MetricsBase):

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
        tf.TensorSpec([], 'int32', name='rec_idx'),
        tf.TensorSpec([None], name='ref_notes'),
        tf.TensorSpec([None, 321], name='logits'),
        tf.TensorSpec([], name='loss')
    ], autograph=False)
    def update_states_tf_fn(self, rec_idx, ref_notes, logits, loss):

        count_nonzero_fn = MetricsBase.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)

        logits = tf.convert_to_tensor(logits, tf.float32)
        logits.set_shape([None, 321])
        logits = logits[:, 1:] - logits[:, :1]

        vth = self.voicing_threshold
        vth = np.log(vth / (1. - vth))
        vth = tf.convert_to_tensor(vth, tf.float32)

        loss = tf.convert_to_tensor(loss, tf.float32)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_peak_indices = tf.argmax(logits, axis=1, output_type=tf.int32)
        est_peak_logits = tf.gather(logits, axis=1, indices=est_peak_indices, batch_dims=1)
        est_voicing = est_peak_logits > vth
        n_est_voicing = tf.logical_not(est_voicing)
        est_notes = MetricsBase.est_notes_interpolation_fn(est_peak_indices)

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
        tf.TensorSpec([None], tf.int32, name='melody_bins'),
        tf.TensorSpec([None], tf.bool, name='est_voicing')
    ], autograph=False)
    def viterbi_update_states_tf_fn(self, rec_idx, ref_notes, melody_bins, est_voicing):

        count_nonzero_fn = MetricsBase.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)

        melody_bins = tf.convert_to_tensor(melody_bins, tf.int32)
        est_voicing = tf.convert_to_tensor(est_voicing, tf.bool)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        n_est_voicing = tf.logical_not(est_voicing)

        est_notes = MetricsBase.est_notes_interpolation_fn(est_peak_indices=melody_bins)

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
            ref_notes = rec_dict['notes']
            assert len(logits) == len(ref_notes)
            viterbi_method = self.model.config.viterbi_method
            if viterbi_method == 'shaun':
                logits = logits[:, 1:] - logits[:, :1]
            voiced_flags, bins = self.model.config.acoustic_model_ins.viterbi(logits)
            est_notes_with_voicing_info = self.viterbi_update_states_tf_fn(
                rec_idx=self.rec_idx,
                ref_notes=ref_notes,
                melody_bins=bins,
                est_voicing=voiced_flags
            )
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

        assert inferencing
        self.name = name
        self.is_training = True if 'train' in name else False
        self.config = config
        self.tf_dataset = TFDatasetForInferenceMode(self)
        self.metrics = MetricsInference(self)
        self.tb_summary_ins = TBSummary(self)


def main():

    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    for name in MODEL_DICT['config'].model_names:
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    aug_info = []
    aug_info.append('viterbi - {}'.format(MODEL_DICT['config'].viterbi_method))
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
            optimizer.apply_gradients(zip(grads, trainable_vars))
            metrics.update_states(ref_notes=batch['notes'][0], logits=logits, loss=loss)
        write_tb_summary_fn(global_step)

        loss = model.metrics.loss
        oa = model.metrics.oa
        logging.info('{} - step - {} - loss - {} - oa - {}'.format(model.name, global_step, loss, oa))

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
        batches_per_epoch = model.tf_dataset.batches_per_epoch
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
        logging.info('{} - step - {} - loss - {} - oa - {}'.format(model.name, global_step, loss, oa))

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

