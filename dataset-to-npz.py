""" Load the dataset from h5 files into the npz format. """

import h5py
import numpy as np
from sklearn.model_selection import train_test_split


files = ['dataset/jetImage_7_100p_30000_40000.h5',
         'dataset/jetImage_7_100p_60000_70000.h5',
         'dataset/jetImage_7_100p_50000_60000.h5',
         'dataset/jetImage_7_100p_10000_20000.h5',
         'dataset/jetImage_7_100p_0_10000.h5']

target = np.array([])
features = np.array([])
for fileIN in files:
    print("Appending %s" % fileIN)
    f = h5py.File(fileIN)
    myFeatures = np.array(f.get("jets")[:, [12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]])
    mytarget = np.array(f.get('jets')[0:, -6:-1])
    features = np.concatenate([features, myFeatures], axis=0) if features.size else myFeatures
    target = np.concatenate([target, mytarget], axis=0) if target.size else mytarget
print(features.shape, target.shape)

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.33)

np.savez('x_test.npz', arr_0=X_val)
np.savez('y_test.npz', arr_0=y_val)
