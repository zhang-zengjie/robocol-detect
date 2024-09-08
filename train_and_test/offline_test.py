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


def plot_confusion_matrix(cmt):
    plt.rcParams.update({'font.size': 12})
    classes = ('contacts', 'free', 'collisions')
    # title = 'Confusion matrix of the model test'
    cmn = cmt.astype('float') / cmt.sum(axis=1)[:, np.newaxis] * 100
    plt.imshow(cmn, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=45)
    thresh = cmn.max() / 2.
    for i, j in itertools.product(range(cmt.shape[0]), range(cmt.shape[1])):
        plt.text(j, i, format(cmt[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="bottom",
                 color="white" if cmn[i, j] > thresh else "black")
        if cmn[i, j] > 10:
            plt.text(j, i, '(' + format(cmn[i, j], '.1f') + '%)',
                     horizontalalignment="center",
                     verticalalignment="top",
                     color="white" if cmn[i, j] > thresh else "black")
        else:
            plt.text(j, i, '(' + format(cmn[i, j], '.2f') + '%)',
                     horizontalalignment="center",
                     verticalalignment="top",
                     color="white" if cmn[i, j] > thresh else "black")
    plt.ylabel('true labels')
    plt.xlabel('predicted labels')
    plt.tight_layout()
    plt.show()

'''
Select the robot joints
'''
# joints = list(range(1, 8))
joints = [1,2,4]

'''
Select the features
'''
# ftrs = np.arange(1, 19)
# ftrs = np.array([3, 6, 8, 12, 14, 15, 17])
# ftrs = np.array([1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])
ftrs = np.array([1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])

tdt = np.loadtxt("dataset/full_set.csv", dtype='float64', delimiter=",")

x_train, x_test, y_train, y_test = shuffle_and_split(tdt)

fs = np.array([0])
for it_n in joints:
    fs = np.concatenate((fs, ftrs+(it_n-1)*18))
fs = fs[1:]

np.savetxt("dataset/x_test.csv", x_test[0:, fs-1], delimiter=",")

cvn = 10

'''
Select a model
'''
# cr = LinearDiscriminantAnalysis(solver='svd')
# cr = GaussianNB()
# cr = SVC(kernel='linear', C=63.1)
# cr = RandomForestClassifier(n_estimators=480, random_state=0)
# cr = neighbors.KNeighborsClassifier(n_neighbors=6, metric='canberra')
cr = MLPClassifier(hidden_layer_sizes=(48,), activation='logistic', random_state=0)

rs = cross_val_score(cr, x_train[0:, fs-1], y_train, cv=cvn)
print(np.mean(rs))
# print(np.var(rs))

cr.fit(x_train[0:, fs-1], y_train)
y_pred = cr.predict(x_test[0:, fs-1])

print(accuracy_score(y_test, y_pred))
plot_confusion_matrix(confusion_matrix(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
