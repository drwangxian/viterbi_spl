import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True)
x = range(4)
ax0.scatter(x, [0, 3, 2, 4], c='k')
ax0.set_title('ax0')
ax0.set_xticks([])
ax0.set_yticks([])
ax1.scatter(x, [0, 3, 2, 4], label='ax1')
ax1.set_xticks([])
ax1.set_yticks([])
ax2.scatter(x, [0, 3, 2, 4], label='ax2')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('time')
fig.suptitle('overall')


plt.savefig('subplots.png')
plt.close()
