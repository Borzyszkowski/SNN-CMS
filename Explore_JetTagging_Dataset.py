""" Explore the Jet Tagging Dataset and select correct features """

import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm

data_path = './dataset'


def load_data(data_folder):
    # Get folder path containing text files
    file_list = glob.glob(data_folder + '/*.h5')
    data = []
    for file_path in file_list:
        data.append(file_path)
    print(f'dataset contains {len(data)} files')
    return data


# this function makes the histogram of a given quantity for the five classes
def disp_plot(feature_index, input_data, input_featurenames, data_representation='hl_features'):
    plt.subplots()
    for particle in range(len(labelCat)):
        # high level features
        if data_representation == 'hl_features':
            # notice the use of numpy masking to select specific classes of jets
            my_data = input_data[np.argmax(target, axis=1) == particle]
            # then plot the right quantity for the reduced array
            plt.hist(my_data[:, feature_index], 50, density=True, histtype='step', fill=False, linewidth=1.5)
        # image representation of data
        elif data_representation == 'images':
            my_data = input_data[:, :, feature_index]
            # notice the use of numpy masking to select specific classes of jets
            my_data = my_data[np.argmax(target, axis=1) == particle]
            # then plot the right quantity for the reduced array
            plt.hist(my_data[:, feature_index].flatten(), 50, density=True, histtype='step', fill=False, linewidth=1.5)
        else:
            return
    plt.yscale('log', nonposy='clip')
    plt.legend(labelCat, fontsize=12, frameon=False)
    plt.xlabel(str(input_featurenames[feature_index], "utf-8"), fontsize=15)
    plt.ylabel('Prob. Density (a.u.)', fontsize=15)
    plt.show()


def image_data(targ, labels):
    image = np.array(file.get('jetImage'))
    image_g = image[np.argmax(targ, axis=1) == 0]
    image_q = image[np.argmax(targ, axis=1) == 1]
    image_W = image[np.argmax(targ, axis=1) == 2]
    image_Z = image[np.argmax(targ, axis=1) == 3]
    image_t = image[np.argmax(targ, axis=1) == 4]
    images = [image_q, image_g, image_W, image_Z, image_t]
    # for i in range(len(images)):
    i = 0
    SUM_Image = np.sum(images[i], axis=0)
    plt.imshow(SUM_Image / float(images[i].shape[0]), origin='lower', norm=LogNorm(vmin=0.01))
    plt.colorbar()
    plt.title(labels[i], fontsize=15)
    plt.xlabel("$\Delta\eta$ cell", fontsize=15)
    plt.ylabel("$\Delta\phi$ cell", fontsize=15)
    plt.show()


def list_data(file):
    print("Particle feature names:")
    p_featurenames = file.get("particleFeatureNames")
    print(p_featurenames[:])

    print("Shape of list")
    p_data = file.get("jetConstituentList")
    print(p_data.shape)

    # plot all the features
    # for i in range(len(p_featurenames)):
    disp_plot(0, p_data, p_featurenames, 'images')


if __name__ == "__main__":
    loaded_data = load_data(data_path)

    print('structure of a single data file:')
    f = loaded_data[0]
    file = h5py.File(f)
    print(list(file.keys()))

    print('names of jets:')
    featurenames = file.get('jetFeatureNames')
    print(featurenames[:])

    jet_data = np.array(file.get('jets'))
    target = jet_data[:, -6:-1]

    labelCat = ["gluon", "quark", "W", "Z", "top"]

    print("dataset shape:")
    data = np.array(jet_data[:, :])
    print(data.shape)

    print("targets shape:")
    print(target.shape)

    print("features shape:")
    features = np.array(jet_data[:, :-6])
    print(features.shape)

    print("selected target classes:")
    print(labelCat)
    print(featurenames[-6:-1])

    """ The physics motivated high level features """
    # plot all the features
    # for i in range(len(featurenames[:-6])):
    disp_plot(0, data, featurenames, 'hl_features')

    """ The image representation of particles """
    image_data(target, labelCat)

    """ The particle list dataset """
    list_data(file)
