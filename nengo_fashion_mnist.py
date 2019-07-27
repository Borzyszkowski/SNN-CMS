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
# import tensorflow.contrib.slim as slim;

import nengo
import nengo_dl

# keras uses the global random seeds, so we set those here to
# ensure the example is reproducible
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = (
    fashion_mnist.load_data())
num_classes = np.unique(test_labels).shape[0]

# normalize images so values are between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(3):
    plt.figure()
    plt.imshow(test_images[i], cmap="gray")
    plt.axis("off")
    plt.title(class_names[test_labels[i]]);


image_shape = (28, 28)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=image_shape, name='flatten'),
    keras.layers.Dense(128, activation=tf.nn.relu, name='hidden'),
    keras.layers.Dense(num_classes, activation=tf.nn.softmax,
                       name='softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5);


model_weights = "keras_weights.h5"
model.save_weights(model_weights)


class KerasNode:
    def __init__(self, keras_model):
        self.model = keras_model

    def pre_build(self, *args):
        self.model = keras.models.clone_model(self.model)

    def __call__(self, t, x):
        # reshape the flattened images into their 2D shape
        # (plus the batch dimension)
        images = tf.reshape(x, (-1,) + image_shape)
        # build the rest of the model into the graph
        return self.model.call(images)

    def post_build(self, sess, rng):
        self.model.load_weights(model_weights)


with tf.Session():
    model.load_weights(model_weights)

    # model.call takes a Tensor as input and returns a Tensor
    out1 = model.call(tf.convert_to_tensor(test_images[:10],
                                           dtype=tf.float32))
    print("Type of 'out1':", type(out1))

    # model.predict takes a numpy array as input and returns
    # a numpy array
    out2 = model.predict(test_images[:10])
    print("Type of 'out2':", type(out2))

net_input_shape = np.prod(image_shape)  # because input will be a vector

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
        size_out=num_classes)

    # connect up our input to our keras node
    nengo.Connection(input_node, keras_node, synapse=None)

    # add a probes to collect output of keras node
    keras_p = nengo.Probe(keras_node)

minibatch_size = 20

# pick some random images from test set
np.random.seed(1)
test_inds = np.random.randint(low=0, high=test_images.shape[0],
                              size=(minibatch_size,))
test_inputs = test_images[test_inds]

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
    plt.imshow(test_images[test_inds][i], cmap="gray")
    plt.axis("off")
    plt.title("%s (%s)" % (
        class_names[test_labels[test_inds][i]],
        class_names[np.argmax(tensornode_output[i, 0])]));
plt.show()
