from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from libs.cdi_lib import shuffle_and_split, mesh_validate

rdt = np.loadtxt("dataset/full_set.csv", dtype='float64', delimiter=",")
x_train, x_test, y_train, y_test = shuffle_and_split(rdt)

joints = [1, 2, 4]
# ftrs = np.array([1, 3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18])
ftrs = np.array([1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])


kernel = ['linear', 'rbf', 'sigmoid']
c = np.arange(-3, 3.01, 0.2)

try:
    dt = np.loadtxt("dataset/validate-svm-data.csv", dtype='float64', delimiter=",")
except IOError:
    print("Validated data file does not exist. Generation starts...")
    fs = np.array([0])
    for it_n in joints:
        fs = np.concatenate((fs, ftrs+(it_n-1)*18))
    fs = fs[1:]
    dt = mesh_validate(x_train[0:, fs-1], y_train, 10, 'svm', kernel, c)
    np.savetxt("dataset/validate-svm-data.csv", dt, delimiter=",")

(rs_mean, rs_var) = np.split(dt, 2, axis=0)
max_idx = np.unravel_index(np.argmax(rs_mean, axis=None), rs_mean.shape)
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(4, 3.6))
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(c, np.arange(0, len(kernel)))
ax.plot_surface(x, y, rs_mean, rstride=1, cstride=1, cmap=cm.coolwarm)

ax.set_yticks(np.arange(1, 3, 1))
ax.set_xlabel(r'$C$ value')
ax.set_xticklabels((r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'1', r'10', r'$10^2$', r'$10^3$'))
ax.set_ylabel('kernel', horizontalalignment='center', verticalalignment='top', labelpad=10)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(('linear', 'rbf', 'sigmoid'), horizontalalignment='left', verticalalignment='top')
ax.set_zlim((0.49, 1.01))
ax.set_zticks(np.arange(0.5, 1.01, 0.1))
ax.set_zticklabels(('50%', '60%', '70%', '80%', '90%', '100%'), horizontalalignment='left')
ax.set_zlabel('Accuracy', horizontalalignment='right')
ax.tick_params(axis='both', which='major', pad=-5)

print("Best kernel: " + kernel[max_idx[0]])
print("Best c value: " + format(c[max_idx[1]], '.1e'))
print("Best score: " + format(rs_mean[max_idx], '.7f'))
print("Best variance: " + format(rs_var[max_idx], '.7e'))

plt.subplots_adjust(left=0, right=0.9, top=0.99, bottom=0.135)
plt.show()
