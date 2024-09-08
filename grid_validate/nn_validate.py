from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from libs.cdi_lib import shuffle_and_split, mesh_validate

joints = [1, 2, 4]
# ftrs = np.array([1, 3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18])
ftrs = np.array([1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])
max_n = 100
neuron = np.arange(3, max_n+1, 1)
activ = ['identity', 'logistic', 'tanh', 'relu']

try:
    dt = np.loadtxt("dataset/validate-nn-data.csv", dtype='float64', delimiter=",")
except IOError:
    print("Validated data file does not exist. Generation starts...")
    rdt = np.loadtxt("dataset/full_set.csv", dtype='float64', delimiter=",")
    x_train, x_test, y_train, y_test = shuffle_and_split(rdt)
    fs = np.array([0])
    for it_n in joints:
        fs = np.concatenate((fs, ftrs+(it_n-1)*18))
    fs = fs[1:]
    dt = mesh_validate(x_train[0:, fs-1], y_train, 10, 'nn', neuron, activ)
    np.savetxt("dataset/validate-nn-data.csv", dt, delimiter=",")

(rs_mean, rs_var) = np.split(dt, 2, axis=0)
max_idx = np.unravel_index(np.argmax(rs_mean, axis=None), rs_mean.shape)
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(4.2, 3.6))
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(0, len(activ)), neuron)
ax.plot_surface(x[2:, 0:], y[2:, 0:], rs_mean[2:, 0:], rstride=1, cstride=1, cmap=cm.coolwarm)

ax.set_zlim((0.949, 0.971))
ax.set_xticks(np.arange(0, 4, 1))
ax.set_zticks(np.arange(0.95, 0.971, 0.01))
ax.set_zticklabels(('95%', '96%', '97%'), horizontalalignment='left')
ax.set_xticklabels(activ, horizontalalignment='right')
ax.set_ylabel(r'# neurons')
ax.set_xlabel(r'Activation function', verticalalignment='top')
ax.set_zlabel('Accuracy', labelpad=10)
ax.tick_params(axis='both', which='major', pad=-5)

print("Best neuron number: " + format(neuron[max_idx[0]]))
print("Best active function: " + activ[max_idx[1]])
print("Best score: " + format(rs_mean[max_idx], '.7f'))
print("Best variance: " + format(rs_var[max_idx], '.7e'))

plt.subplots_adjust(left=0.05, right=0.9, top=0.99, bottom=0.135)
plt.show()
