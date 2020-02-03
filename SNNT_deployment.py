""" Script to train and run a neural network on Loihi to solve Jet Tagging Task, using SNN toolbox """

import configparser
from snntoolbox.bin.run import main
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.model_selection import train_test_split
import glob

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
    print('input_shape: ', input_shape)

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

    batch_size = 128
    n_epochs = 50

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

    # save model with weights and optimizer
    model.save('model2SNN.h5')
    print('model exported')

    # save weights
    model.save_weights("./model_files/jet_file.ckpt")

    # save model
    model_json = model.to_json()
    with open(json_file, "w") as jf:
        jf.write(model_json)


def create_config():
    config = configparser.ConfigParser()

    config['paths'] = {
        'path_wd': '.',                 # Path to model.
        'dataset_path': data_path,      # Path to dataset.
        'filename_ann': 'model2SNN',    # Name of input model.
        'runlabel': 'test'
    }

    config['tools'] = {
        'evaluate_ann': True,           # Test ANN on dataset before conversion.
        'parse': True,                  # Parses input model.
        'normalize': False              # Not needed for Loihi backend.
    }

    config['simulation'] = {
        'simulator': 'loihi',           # Chooses execution backend of SNN toolbox.
        'duration': 3000,               # Number of time steps to run each sample.
        'num_to_test': 10,              # How many samples to run.
        'batch_size': 1,                # Batch size 1 is the only supported value.
        'keras_backend': 'tensorflow'}

    config['loihi'] = {
        # Set partition to run on (optional).
        'partition': os.environ.get('PARTITION', 'loihi'),
        # Save plots about resource utilization.
        'save_output': True,
        # Accelerate reset between samples.
        'use_reset_snip': True,
        # Estimate overflow of dendritic accumulator.
        'do_overflow_estimate': True,
        # Tune thresholds to optimal dynamic range.
        'normalize_thresholds': True,
        # Hyperparameter for threshold normalization. Defines margin between
        # threshold and maximum net input to a neuron. Larger ratio -> lower
        # quantization error, but higher runtime needed.
        'desired_threshold_to_input_ratio': 2**7,
        # For DNNs it is recommended to set the biasExp = 6 and weightExponent = 0.
        'compartment_kwargs': {'biasExp': 6, 'vThMant': 2 ** 9},
        'connection_kwargs': {'numWeightBits': 8, 'weightExponent': 0}
    }

    config['output'] = {
        'log_vars': {'all'},
        'plot_vars': {'all'}
    }

    config['input'] = {
        'model_lib': 'keras',
        'dataset_format': 'npz'
    }

    config_path = 'conversion_config_loihi.txt'
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    return config_path


if __name__ == "__main__":
    data = load_data(data_path)
    make_model(data, json_path, h5_path)
    config_filepath = create_config()
    main(config_filepath)
