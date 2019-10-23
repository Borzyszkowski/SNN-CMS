""" Trains the model and saves it to the json format with weights in h5 format. """

import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.model_selection import train_test_split

from Inputs_to_h5 import gen_h5
import glob
import tensorflow as tf

# give paths to the dataset folder and json + h5 files
data_path = './dataset'
json_path = "model.json"
h5_path = "weights.h5"


def load_data(data_folder):
    print("loading the dataset")
    # Get folder path containing text files
    file_list = glob.glob(data_folder + '/*.h5')
    data = []
    for file_path in file_list:
        data.append(file_path)
    print("dataset loaded")
    return data


def make_model(files, json_file, h5_file):
    # preparing the dataset
    target = np.array([])
    features = np.array([])
    for fileIN in files:
        # print("Appending %s" % fileIN)
        f = h5py.File(fileIN)
        myFeatures = np.array(f.get("jets")[:, [12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]])
        mytarget = np.array(f.get('jets')[0:, -6:-1])
        features = np.concatenate([features, myFeatures], axis=0) if features.size else myFeatures
        target = np.concatenate([target, mytarget], axis=0) if target.size else mytarget

    x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.33)

    np.savez('x_test.npz', arr_0=features)
    np.savez('y_test.npz', arr_0=target)
    print("npz dataset files created")

    # creating the model
    input_shape = x_train.shape[1]
    print(input_shape)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    inputArray = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(inputArray)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(32, activation='relu')(x)

    output = Dense(5, activation='softmax')(x)
    model = Model(inputs=inputArray, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model_json = model.to_json()
    with open(json_file, "w") as jf:
        jf.write(model_json)
    print("json model saved")

    batch_size = 128
    n_epochs = 50

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        # training the model
        history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=2,
                            validation_data=(x_val, y_val),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                                       ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),
                                       TerminateOnNaN()])

        # visualizing the history
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.yscale('log')
        plt.title('Training History')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('ANN_training.png')
        plt.show()
        model.save_weights(h5_file)
        print("h5 weight file saved")
        saver.save(sess, "./model_files/jet_file.ckpt")
        print('model files saved: meta, index and data')


if __name__ == "__main__":
    data = load_data(data_path)
    make_model(data, json_path, h5_path)
    gen_h5(json_path, h5_path)
