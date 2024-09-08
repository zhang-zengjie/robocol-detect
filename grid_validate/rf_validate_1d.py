import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from libs.cdi_lib import shuffle_and_split
from matplotlib import pyplot as plt
import itertools

joints = list(range(1, 8))
# ftrs = np.array([1, 3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18])
ftrs = np.array([1, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18])
mf = np.arange(6, 126, 4)

try:
    mn = np.loadtxt("dataset/validate-rf-data-mtry-mean.csv", dtype=np.float, delimiter=",")
    vr = np.loadtxt("dataset/validate-rf-data-mtry-var.csv", dtype=np.float, delimiter=",")
except IOError:
    print("Validated data file does not exist. Generation starts...")
    rdt = np.loadtxt("dataset/full_set.csv", dtype=np.float, delimiter=",")
    x_train, x_test, y_train, y_test = shuffle_and_split(rdt)
    print("Data prepared...")
    mn = np.empty(mf.shape[0], dtype=float)
    vr = np.empty(mf.shape[0], dtype=float)
    cvn = 10
    fs = np.array([0])
    for it_n in joints:
        fs = np.concatenate((fs, ftrs + (it_n-1)*18))
    fs = fs[1:]
    print("feature generated...")
    for j, it_n in enumerate(mf):
        cr = RandomForestClassifier(n_estimators=480, max_features=it_n, max_depth=26, random_state=0)
        rs = cross_val_score(cr, x_train[0:, fs - 1], y_train, cv=cvn)
        mn[j] = np.mean(rs)
        vr[j] = np.var(rs)
        print("Training in process: " + format(float(it_n) * 100 / float(126), '.2f') + '%...')
    np.savetxt("dataset/validate-rf-data-mtry-mean.csv", mn, delimiter=",")
    np.savetxt("dataset/validate-rf-data-mtry-var.csv", vr, delimiter=",")

max_idx = np.unravel_index(np.argmax(mn, axis=None), mn.shape)
plt.rcParams.update({'font.size': 14})
fig = plt.figure()
plt.plot(mf, mn)

print("Best depth: " + format(mf[max_idx[0]]))
print("Best score: " + format(mn[max_idx], '.7f'))
print("Best variance: " + format(vr[max_idx], '.7e'))
