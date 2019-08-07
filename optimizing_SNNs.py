
import gzip
import pickle
from urllib.request import urlretrieve
import zipfile
import requests

import nengo
import nengo_dl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
with gzip.open("mnist.pkl.gz") as f:
    train_data, _, test_data = pickle.load(f, encoding="latin1")
train_data = list(train_data)
test_data = list(test_data)
for data in (train_data, test_data):
    one_hot = np.zeros((data[0].shape[0], 10))
    one_hot[np.arange(data[0].shape[0]), data[1]] = 1
    data[1] = one_hot

for i in range(3):
    plt.figure()
    plt.imshow(np.reshape(train_data[0][i], (28, 28)))
    plt.axis('off')
    plt.title(str(np.argmax(train_data[1][i])));


# lif parameters
lif_neurons = nengo.LIF(amplitude=0.01)

# softlif parameters (lif parameters + sigma)
softlif_neurons = nengo_dl.SoftLIFRate(amplitude=0.01, sigma=0.001)

# ensemble parameters
ens_params = dict(max_rates=nengo.dists.Choice([100]), intercepts=nengo.dists.Choice([0]))

# plot some example LIF tuning curves
for neuron_type in (lif_neurons, softlif_neurons):
    with nengo.Network(seed=0) as net:
        ens = nengo.Ensemble(10, 1, neuron_type=neuron_type)

    with nengo_dl.Simulator(net) as sim:
        plt.figure()
        plt.plot(*nengo.utils.ensemble.tuning_curves(ens, sim))
        plt.xlabel("input value")
        plt.ylabel("firing rate")
        plt.title(str(neuron_type))



def build_network(neuron_type, output_synapse=None):
    with nengo.Network() as net:
        # we'll make all the nengo objects in the network
        # non-trainable. we could train them if we wanted, but they don't
        # add any representational power so we can save some computation
        # by ignoring them. note that this doesn't affect the internal
        # components of tensornodes, which will always be trainable or
        # non-trainable depending on the code written in the tensornode.
        nengo_dl.configure_settings(trainable=False)

        # the input node that will be used to feed in input images
        inp = nengo.Node([0] * 28 * 28)

        # add the first convolutional layer
        x = nengo_dl.tensor_layer(
            inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
            kernel_size=3)

        # apply the neural nonlinearity
        x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

        # add another convolutional layer
        x = nengo_dl.tensor_layer(
            x, tf.layers.conv2d, shape_in=(26, 26, 32),
            filters=64, kernel_size=3)
        x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

        # add a pooling layer
        x = nengo_dl.tensor_layer(
            x, tf.layers.average_pooling2d, shape_in=(24, 24, 64),
            pool_size=2, strides=2)

        # another convolutional layer
        x = nengo_dl.tensor_layer(
                x, tf.layers.conv2d, shape_in=(12, 12, 64),
                filters=128, kernel_size=3)
        x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

        # another pooling layer
        x = nengo_dl.tensor_layer(
            x, tf.layers.average_pooling2d, shape_in=(10, 10, 128),
            pool_size=2, strides=2)

        # linear readout
        x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)

        p = nengo.Probe(x, synapse=output_synapse)

    return net, inp, p

# construct the network
net, inp, out_p = build_network(softlif_neurons)

# construct the simulator
minibatch_size = 200
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)


# note that we need to add the time dimension (axis 1), which has length 1
# in this case. we're also going to reduce the number of test images, just to
# speed up this example.
train_inputs = {inp: train_data[0][:, None, :]}
train_targets = {out_p: train_data[1][:, None, :]}
test_inputs = {inp: test_data[0][:minibatch_size*2, None, :]}
test_targets = {out_p: test_data[1][:minibatch_size*2, None, :]}


def objective(x, y):
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=y)


opt = tf.train.RMSPropOptimizer(learning_rate=0.001)


def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


# print("error before training: %.2f%%" % sim.loss(test_inputs, test_targets,
#                                                  classification_error))

do_training = False
if do_training:
    # run training
    sim.train(train_inputs, train_targets, opt, objective=objective, n_epochs=10)

    # save the parameters to file
    sim.save_params("./mnist_params")
else:
    # download pretrained weights
    urlretrieve(
        "https://drive.google.com/uc?export=download&id=1tNdnMPotIk8N0_rdv0XaL24dYqMGC2iM",
        "mnist_params.zip")
    with zipfile.ZipFile("mnist_params.zip") as f:
        f.extractall()

    # load parameters
    sim.load_params("./mnist_params")

print("error after training: %.2f%%" % sim.loss(test_inputs, test_targets,
                                                classification_error))

sim.close()


net, inp, out_p = build_network(lif_neurons, output_synapse=0.1)

sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size, unroll_simulation=10)
sim.load_params("./mnist_params")

n_steps = 30
test_inputs_time = {inp: np.tile(v, (1, n_steps, 1)) for v in test_inputs.values()}
test_targets_time = {out_p: np.tile(v, (1, n_steps, 1)) for v in test_targets.values()}

print("spiking neuron error: %.2f%%" % sim.loss(test_inputs_time, test_targets_time,
                                                classification_error))

sim.run_steps(n_steps, input_feeds={inp: test_inputs_time[inp][:minibatch_size]})

for i in range(5):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.reshape(test_data[0][i], (28, 28)))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(sim.trange(), sim.data[out_p][i])
    plt.legend([str(i) for i in range(10)], loc="upper left")
    plt.xlabel("time")


sim.close()