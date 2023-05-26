import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



track_name = 'MatthewEntwistle_Lontano'
npz_file_name = 'shaun_' + track_name + '.npz'

npz_data_dict = np.load(npz_file_name)
# voicing = npz_data_dict['no_viterbi_est_voicing']
# est_notes = npz_data_dict['no_viterbi_est_notes']
# n_frames = len(est_notes)
# assert len(voicing) == len(est_notes)
# est_notes[np.logical_not(voicing)] = np.nan
#
# fig = go.Figure()
# x = np.arange(n_frames)
# t = go.Scatter(x=x, y=est_notes, mode='markers')
# fig.add_trace(t)
# fig.update_traces(marker=dict(size=3))
# fig.write_html('{}.html'.format(track_name), auto_open=False)

start_n = 13322
end_n = 15040

start_n = start_n * 441 // 256
end_n = end_n * 441 // 256

est_voicing = npz_data_dict['no_viterbi_est_voicing']
est_notes = npz_data_dict['no_viterbi_est_notes']
ref_notes = npz_data_dict['ref_notes']
viterbi_voicing = npz_data_dict['viterbi_est_voicing']
viterbi_notes = npz_data_dict['viterbi_est_notes']
assert len(ref_notes) == len(est_notes) == len(est_voicing) == len(viterbi_voicing) == len(viterbi_notes)

est_voicing = est_voicing[start_n:end_n]
est_notes = est_notes[start_n:end_n]
ref_notes = ref_notes[start_n:end_n]
viterbi_voicing = viterbi_voicing[start_n:end_n]
viterbi_notes = viterbi_notes[start_n:end_n]

est_notes[np.logical_not(est_voicing)] = np.nan
ref_notes[ref_notes == 0] = np.nan
viterbi_notes[np.logical_not(viterbi_voicing)] = np.nan

fig, (ax_ref, ax_no, ax_viterbi) = plt.subplots(3, sharex=True)
n_frames = len(est_voicing)
x = np.arange(n_frames)
for ax, data, name in zip((ax_ref, ax_no, ax_viterbi), (ref_notes, est_notes, viterbi_notes), ('ref', 'est', 'viterbi')):
    ax.scatter(x=x, y=data, s=.5, c='k')
    ax.set_ylabel(name)
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig('{}_snippet.png'.format(track_name))
plt.close()







