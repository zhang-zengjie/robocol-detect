import numpy as np
from libs.cdi_lib import shuffle_and_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import itertools
from scipy.io import loadmat

joints = list(range(1, 8))
ftrs = np.array([1, 3, 4, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18])
fs = np.array([0])
for it_n in joints:
    fs = np.concatenate((fs, ftrs + (it_n - 1) * 18))
fs = fs[1:]
print("Training the classifier...")
tdt = np.loadtxt("dataset/full_set.csv", dtype='float64', delimiter=",")
x_train, x_test, y_train, y_test = shuffle_and_split(tdt)
cr = RandomForestClassifier(n_estimators=480, max_depth=26, random_state=0)
cr.fit(x_train[0:, fs-1], y_train)



d1 = loadmat('dataset/JK_MsrTrq.mat')
sg = d1['MsrTrq']
d2 = loadmat('dataset/DataStillSpl.mat')
cp = d2['DataStillSpl']

szsg = sg.shape[1]

ydt = np.zeros(szsg, dtype=int)
rdt = np.zeros(szsg, dtype=int)

for it_n in np.arange(500, sg.shape[1]-1000):
    ydt[it_n] = cr.predict(sg[1:8, it_n-300:it_n]-cp[1:8, 200:500])
    print("Predicting samples" + format(float(it_n)*100/float(sg.shape[1]-1500), '.2f') + '%...')

for ip in np.arange(50, 160000, 1):
    if ((ydt[ip - 5:ip] == 1).sum() > 2):
        rdt[ip] = 1
    elif ((ydt[ip - 7:ip] == -1).sum() > 3):
        rdt[ip] = -1
    else:
        rdt[ip] = 0

plt.plot(sg[0, 0:]/1000, sg[1:8, 0:], color='#ff6666', linewidth=2, label=r'Residual $\tau^*_e$')


plt.axis([44.5, 75.5, -0.01, 1.51]) #
plt.ylabel(r'Measured torques ($N \cdot m$)')
plt.xlabel('Time(s)')
fig = plt.gcf()
fig.subplots_adjust(bottom=0.2)
plt.bar(sg[0, 0:]/1000, 4 * rdt, width=0.1, color='#d1d1e0', label=r'Collisions')
plt.legend(loc='upper center', ncol=3)


plt.gcf().subplots_adjust(bottom=0.15)
plt.show()