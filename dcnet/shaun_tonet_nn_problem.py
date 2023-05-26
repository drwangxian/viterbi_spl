import os.path
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


track_name = 'MatthewEntwistle_Lontano'

start_n_441 = 13322
end_n_441 = 15040

tonet_npz_file = 'tonet_' + track_name + '.npz'
tonet_npz_file = os.path.join('../tonet_post_processing_by_viterbi', tonet_npz_file)
shaun_npz_file = 'shaun_' + track_name + '.npz'

tonet_data_dict = np.load(tonet_npz_file)
tonet_ref_notes = tonet_data_dict['ref_notes'][start_n_441:end_n_441]
tonet_ref_voicing = tonet_ref_notes > 0
tonet_ref_notes[np.logical_not(tonet_ref_voicing)] = np.nan
tonet_est_notes = tonet_data_dict['no_viterbi_est_notes']
tonet_est_voicing = tonet_data_dict['no_viterbi_est_voicing']
tonet_est_notes = tonet_est_notes[start_n_441:end_n_441]
tonet_est_voicing = tonet_est_voicing[start_n_441:end_n_441]
tonet_est_notes[np.logical_not(tonet_est_voicing)] = np.nan

start_n_256 = start_n_441 * 441 // 256
end_n_256 = end_n_441 * 441 // 256
shaun_data_dict = np.load(shaun_npz_file)
shaun_est_voicing = shaun_data_dict['no_viterbi_est_voicing']
shaun_est_notes = shaun_data_dict['no_viterbi_est_notes']
assert len(shaun_est_notes) == len(shaun_est_voicing)
shaun_est_voicing = shaun_est_voicing[start_n_256:end_n_256]
shaun_est_notes = shaun_est_notes[start_n_256:end_n_256]
shaun_est_notes[np.logical_not(shaun_est_voicing)] = np.nan

n_frames_441 = len(tonet_ref_notes)
times_441 = np.arange(n_frames_441) * 0.01 + start_n_441 * 0.01
n_frames_256 = len(shaun_est_voicing)
times_256 = np.arange(n_frames_256) * 256. / 44100 + start_n_256 * 256. / 44100

fig, (ax_ref, ax_tonet, ax_shaun) = plt.subplots(3, sharex=True)
for ax, x, y, name in zip(
        (ax_ref, ax_tonet, ax_shaun),
        (times_441, times_441, times_256),
        (tonet_ref_notes, tonet_est_notes, shaun_est_notes),
        ('reference', 'TONET', 'DCNET')):
    ax.scatter(x=x, y=y, s=.5, c='k')
    ax.set_ylabel(name)
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig('tonet_shaun_{}_snippet.pdf'.format(track_name), bbox_inches='tight')
plt.close()





