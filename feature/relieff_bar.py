import matlab.engine
from matplotlib import pyplot as plt
import numpy as np
from libs.cdi_lib import shuffle_and_split

n_joint = 7
n_ftr = 18

try:
    dt = np.loadtxt("dataset/relieff_set.csv", dtype=np.float, delimiter=",")
except IOError:
    print("Validated data file does not exist. Generation starts...")
    tdt = np.loadtxt("dataset/full_set.csv", dtype=np.float, delimiter=",")
    x_train, x_test, y_train, y_test = shuffle_and_split(tdt)
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    print("Engine started...")
    rank, weight = eng.relieff(matlab.double(x_train.tolist()), matlab.double(y_train.tolist()), 100, 'method',
                               'classification', nargout=2)
    eng.quit()
    dt = np.array(weight[0]).reshape((n_joint, n_ftr))
    np.savetxt("dataset/relieff_set", dt, delimiter=",")

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots()
ax.grid(True, linestyle='--')

ax.set_ylim((-0.001, 6.001))
ax.set_ylabel('weight')
ax.set_yticks(np.arange(0, 7, 1))

# If plot according to features:
dt = np.transpose(dt)
ax.set_xlabel('# feature')
# If plot according to joints:
# ax.set_xlabel('# joint')

idx = dt.shape
mdt = dt.reshape((idx[0]*idx[1],))*100.0
ax.set_xlim((0, idx[0]*idx[1]+1))
ax.set_xticks(np.arange(0.5, idx[0]*idx[1], idx[1]))
ax.set_xticklabels(np.arange(1, idx[0]+1))

for it_n in np.arange(0, idx[0], 1):
    av = np.mean(mdt[it_n*idx[1]:(it_n+1)*idx[1]])
    t = plt.text(it_n*idx[1]+idx[1]/2, av+0.13, format(av, '.2f'), color='#000033', va='bottom', ha='center', style='oblique', weight='bold', fontsize=8)
    t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='white'))
    plt.plot([it_n * idx[1] + 1, (it_n + 1) * idx[1]], [av, av], color='#1a1a4c', linewidth=2)


x = np.arange(1, len(mdt)+1)
ax.bar(x, mdt, width=1)
plt.text(0, 6.1, r'$\times 10^{-2}$')
plt.show()
