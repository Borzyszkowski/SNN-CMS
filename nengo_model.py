import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.model_selection import train_test_split
from inputs_to_h5 import gen_h5
import sys
import os
from urllib.request import urlopen
import io
import shutil
import stat

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.slim as slim;

import nengo
import nengo_dl

# keras uses the global random seeds, so we set those here to
# ensure the example is reproducible
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# give paths to the dataset files
dataset_files = ['dataset/jetImage_7_100p_30000_40000.h5',
                 'dataset/jetImage_7_100p_60000_70000.h5',
                 'dataset/jetImage_7_100p_50000_60000.h5',
                 'dataset/jetImage_7_100p_10000_20000.h5',
                 'dataset/jetImage_7_100p_0_10000.h5']

# give paths to the json and h5 files
json_file = "model.json"
h5_file = "weights.h5"


# preparing the dataset
target = np.array([])
features = np.array([])
for fileIN in dataset_files:
    # print("Appending %s" % fileIN)
    f = h5py.File(fileIN)
    myFeatures = np.array(f.get("jets")[:, [12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]])
    mytarget = np.array(f.get('jets')[0:, -6:-1])
    features = np.concatenate([features, myFeatures], axis=0) if features.size else myFeatures
    target = np.concatenate([target, mytarget], axis=0) if target.size else mytarget
# print(features.shape, target.shape)

x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.33)
print("dataset created")

# creating the model
input_shape = x_train.shape[1]
dropoutRate = 0.25

inputArray = Input(shape=(input_shape,))
x = Dense(64, activation='relu')(inputArray)
x = Dropout(dropoutRate)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(dropoutRate)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(dropoutRate)(x)

output = Dense(5, activation='softmax')(x)
model = Model(inputs=inputArray, outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# model_json = model.to_json()
# with open(json_file, "w") as jf:
#     jf.write(model_json)
print("model created")

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
model.save_weights(h5_file)
print("h5 weight file saved")


class KerasNode:
    def __init__(self, keras_model):
        self.model = keras_model

    def pre_build(self, *args):
        self.model = keras.models.clone_model(self.model)

    def __call__(self, t, x):
        # reshape the flattened images into their 2D shape
        # (plus the batch dimension)
        images = tf.reshape(x, (-1,) + input_shape)
        # build the rest of the model into the graph
        return self.model.call(images)

    def post_build(self, sess, rng):
        self.model.load_weights(h5_file)


with tf.Session():
    model.load_weights(h5_file)

    # model.call takes a Tensor as input and returns a Tensor
    out1 = model.call(tf.convert_to_tensor(x_val[:10],
                                           dtype=tf.float32))
    print("Type of 'out1':", type(out1))

    # model.predict takes a numpy array as input and returns
    # a numpy array
    out2 = model.predict(x_val[:10])
    print("Type of 'out2':", type(out2))


net_input_shape = np.prod(input_shape)  # because input will be a vector

with nengo.Network() as net:
    # create a normal input node to feed in our test image.
    # the `np.ones` array is a placeholder, these
    # values will be replaced with the Fashion MNIST images
    # when we run the Simulator.
    input_node = nengo.Node(output=np.ones((net_input_shape,)))

    # create a TensorNode containing the KerasNode we defined
    # above, passing it the Keras model we created.
    # we also need to specify size_in (the dimensionality of
    # our input vectors, the flattened images) and size_out (the number
    # of classification classes output by the keras network)
    keras_node = nengo_dl.TensorNode(
        KerasNode(model),
        size_in=net_input_shape,
        size_out=5)

    # connect up our input to our keras node
    nengo.Connection(input_node, keras_node, synapse=None)

    # add a probes to collect output of keras node
    keras_p = nengo.Probe(keras_node)


minibatch_size = 20

# pick some random images from test set
np.random.seed(1)
test_inds = np.random.randint(low=0, high=x_val.shape[0],
                              size=(minibatch_size,))
test_inputs = x_val[test_inds]

# flatten images so we can pass them as vectors to the input node
test_inputs = test_inputs.reshape((-1, net_input_shape))

# unlike in Keras, NengoDl simulations always run over time.
# so we need to add the time dimension to our data (even though
# in this case we'll just run for a single timestep).
test_inputs = test_inputs[:, None, :]

with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
    sim.step(data={input_node: test_inputs})

tensornode_output = sim.data[keras_p]

for i in range(5):
    plt.figure()
    plt.imshow(x_val[test_inds][i], cmap="gray")
    plt.axis("off")
    plt.title("%s (%s)" % (
        mytarget[target[test_inds][i]],
        mytarget[np.argmax(tensornode_output[i, 0])]));
