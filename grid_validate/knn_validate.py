from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from libs.cdi_lib import shuffle_and_split, mesh_validate

joints = [1, 2, 4]
# ftrs = np.array([1, 3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18])
ftrs = np.array([1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])
max_n = 50
k = np.arange(3, max_n+1, 1)
dm = ('manhattan', 'canberra', 'euclidean', 'minkowski', 'chebyshev')

try:
    dt = np.loadtxt("dataset/validate-knn-data.csv", dtype='float64', delimiter=",")
except IOError:
    print("Validated data file does not exist. Generation starts...")
    rdt = np.loadtxt("dataset/full_set.csv", dtype='float64', delimiter=",")
    x_train, x_test, y_train, y_test = shuffle_and_split(rdt)
    fs = np.array([0])
    for it_n in joints:
        fs = np.concatenate((fs, ftrs+(it_n-1)*18))
    fs = fs[1:]
    dt = mesh_validate(x_train[0:, fs-1], y_train, 10, 'knn', k, dm)
    np.savetxt("dataset/validate-knn-data.csv", dt, delimiter=",")

(rs_mean, rs_var) = np.split(dt, 2, axis=0)
max_idx = np.unravel_index(np.argmax(rs_mean, axis=None), rs_mean.shape)
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(4, 3.6))
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(0, len(dm)), k)
ax.plot_surface(x, y, rs_mean, rstride=1, cstride=1, cmap=cm.coolwarm)

alignment = {'fontsize': 14, 'horizontalalignment': 'right', 'verticalalignment': 'top'}
ax.set_ylim((-1, max_n+1))
ax.set_yticks(np.arange(0, max_n+1, 10))
ax.set_ylabel(r'$k$ value')

ax.set_xticks(np.arange(0, len(dm)))
ax.set_xlabel('distance metrics')
ax.set_xticklabels(('man', 'can', 'euc', 'min', 'che'), fontdict=alignment)

ax.set_zlim((0.749, 1.001))
ax.set_zticks(np.arange(0.75, 1.001, 0.05))
ax.set_zticklabels(('75%', '80%', '85%', '90%', '95%', '100%'), horizontalalignment='left')
ax.set_zlabel('Accuracy', labelpad=8)
ax.tick_params(axis='both', which='major', pad=-5)

print("Best k value: " + format(k[max_idx[0]]))
print("Best distance metrics: " + dm[max_idx[1]])
print("Best score: " + format(rs_mean[max_idx], '.7f'))
print("Best variance: " + format(rs_var[max_idx], '.7e'))

plt.subplots_adjust(left=0, right=0.9, top=0.99, bottom=0.135)
plt.show()
