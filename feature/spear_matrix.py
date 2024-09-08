from libs.cdi_lib import shuffle_and_split
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

n_joint = 7
n_ftr = 18

try:
    dt = np.loadtxt("dataset/spear_data.csv", dtype='float64', delimiter=",")
except IOError:
    print("Data file does not exist. Generation starts...")
    tdt = np.loadtxt("dataset/full_set.csv", dtype='float64', delimiter=",")
    dn = tdt.shape
    x_train, x_test, y_train, y_test = shuffle_and_split(tdt)
    dt = np.zeros((dn[1] - 1, dn[1] - 1))
    for i in np.arange(0, dn[1] - 1, 1):
        for j in np.arange(0, dn[1] - 1, 1):
            rest = spearmanr(x_train[:, i], x_train[:, j])
            dt[i, j] = rest[0]
    np.savetxt("dataset/spear_data.csv", dt, delimiter=",")

s_dt = dt.shape
idx = np.arange(0, s_dt[0])
#idx = np.transpose(np.arange(0, s_dt[0]).reshape((n_joint, n_ftr))).reshape((s_dt[0],))

plt.rcParams.update({'font.size': 12})
dt = dt[::-1, 0:]

plt.imshow(dt[idx][idx], interpolation='nearest', vmin=-1, vmax=1, cmap=plt.cm.coolwarm)
plt.colorbar(ticks=[-1, -0.5, 0, 0.5, 1])
plt.xticks(list(range(-1, dt.shape[0]-1, 18)), list(range(1, 8, 1)))
plt.yticks(list(range(17, dt.shape[0]-1, 18)), list(range(6, 0, -1)))
plt.xlabel('# joint')
plt.ylabel('# joint')
plt.grid(linestyle='-.', color='k', linewidth=1)
plt.show()
