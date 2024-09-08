import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import neighbors


def mesh_validate(x_train, y_train, cvn, model, hp_1, hp_2):
    rs_mean = np.empty((len(hp_1), len(hp_2)))
    rs_var = np.empty((len(hp_1), len(hp_2)))
    for i, it_1 in enumerate(hp_1):
        for j, it_2 in enumerate(hp_2):
            if model == 'rf':
                cl_result = RandomForestClassifier(n_estimators=it_2, max_features=it_1, random_state=0)
            elif model == 'svm':
                cl_result = SVC(C=10**float(it_2), kernel=it_1)
            elif model == 'knn':
                cl_result = neighbors.KNeighborsClassifier(n_neighbors=it_1, metric=it_2)
            else:  # default model set as 'nn'
                cl_result = MLPClassifier(hidden_layer_sizes=(it_1,), activation=it_2, random_state=0, max_iter=400)
            rs_score = cross_val_score(cl_result, x_train, y_train, cv=cvn)
            rs_mean[i, j] = np.mean(rs_score)
            rs_var[i, j] = np.var(rs_score)
            print("Training in process: " + format(float(i*len(hp_2)+j)*100/float(len(hp_1)*len(hp_2)), '.2f') + '%...')
    print("Training complete!")
    return np.vstack((rs_mean, rs_var))


def shuffle_and_split(dt):
    kt = shuffle(dt, random_state=0)
    y, x = kt[0:, 0], kt[0:, 1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)
    return x_train, x_test, y_train, y_test
