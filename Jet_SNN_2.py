""" Solving a Jet Tagging problem with Spiking Neural Networks deployed on the Loihi chip. """

import h5py
import glob
import nengo
import numpy as np
import nengo_dl
import nengo_loihi
import tensorflow as tf
import nengo_extras
import pickle
import gzip
from functools import partial

# give paths to the dataset folder and json + h5 files
data_path = './dataset'
json_path = "model.json"
h5_path = "weights.h5"

print("loading the dataset")
# Get folder path containing text files
file_list = glob.glob(data_path + '/*.h5')
dataset = []
for file_path in file_list:
    dataset.append(file_path)
print("dataset loaded")

# preparing the dataset
target = np.array([])
features = np.array([])
for fileIN in dataset:
    # print("Appending %s" % fileIN)
    f = h5py.File(fileIN)
    myFeatures = np.array(f.get("jets")[:, [12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]])
    mytarget = np.array(f.get('jets')[0:, -6:-1])
    features = np.concatenate([features, myFeatures], axis=0) if features.size else myFeatures
    target = np.concatenate([target, mytarget], axis=0) if target.size else mytarget
print(target.shape, features.shape)

# splitting the train / test data in ratio 80:20
train_features = features[:64000]
train_targets = target[:64000]
test_features = features[64000:]
test_targets = target[64000:]

# creating a train and test dataset
test_d = []
train_d = []
train_d.append(train_features)
train_d.append(train_targets)
test_d.append(test_features)
test_d.append(test_targets)

n_inputs = 16
n_outputs = 5
max_rate = 100
amplitude = 1/max_rate


def crossentropy(outputs, targets):
    """Cross-entropy loss function (for training)."""
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=outputs, labels=targets))


def percentile_l2_loss_range(x, y, weight=1.0, min=0.0, max=np.inf, percentile=99.):
    # x axes are (batch examples, time (==1), neurons)
    assert len(x.shape) == 3
    neuron_p = tf.contrib.distributions.percentile(x, percentile, axis=(0, 1))
    low_error = tf.maximum(0.0, min - neuron_p)
    high_error = tf.maximum(0.0, neuron_p - max)
    return weight * tf.nn.l2_loss(low_error + high_error)


max_rate = 100
amp = 1. / max_rate
rate_reg = 1e-2
rate_target = max_rate * amp  # must be in amplitude scaled units

# model for Jet classification
with nengo.Network(label="Jet classification") as model:
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None

    neuron_type = nengo.SpikingRectifiedLinear(amplitude=amp)
    # neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.001, amplitude=amp)
    # neuron_type = nengo.AdaptiveLIF(amplitude=amp)
    # neuron_type = nengo.Izhikevich()

    u = nengo.Node(np.zeros(16), label="in")
    layer1 = nengo.Ensemble(100, 1, neuron_type=neuron_type)
    layer2 = nengo.Ensemble(120, 1, neuron_type=neuron_type)
    out = nengo.Node(size_in=5)

    nengo.Connection(u, layer1.neurons, transform=1)
    nengo.Connection(layer1.neurons, layer2.neurons, transform=2)
    nengo.Connection(layer2.neurons, out, transform=3)

    layer1_p = nengo.Probe(layer1.neurons)
    layer2_p = nengo.Probe(layer2.neurons)
    out_p = nengo.Probe(out)

objective = {}
objective[out_p] = crossentropy

objective[layer1_p] = partial(
    percentile_l2_loss_range, weight=rate_reg,
    min=0.2*rate_target, max=rate_target, percentile=99.9)

objective[layer2_p] = partial(
    percentile_l2_loss_range, weight=rate_reg,
    min=0.5*rate_target, max=rate_target, percentile=99.0)


def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1), tf.argmax(targets[:, -1], axis=-1)), tf.float32))


dt = 0.001  # simulation timestep
presentation_time = 0.1  # input presentation time
train_data = {u: train_d[0][:, None, :], out_p: train_d[1][:, None, :]}

# for the test data evaluation we'll be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
test_data = {u: np.tile(test_d[0][:, None, :], (1, int(presentation_time / dt), 1)),
             out_p: np.tile(test_d[1][:, None, :], (1, int(presentation_time / dt), 1))}

minibatch_size = 200
do_training = True
with nengo_dl.Simulator(model, minibatch_size=minibatch_size, seed=0) as sim:
    if do_training:
        print("error before training: %.2f%%" % sim.loss(test_data, {out_p: classification_error}))
        # run training
        sim.train(train_data, tf.train.RMSPropOptimizer(learning_rate=0.001),
                  objective={out_p: crossentropy}, n_epochs=20)
        print("error after training: %.2f%%" % sim.loss(test_data, {out_p: classification_error}))
        sim.save_params("./jet_params")
    else:
        print("error before training: %.2f%%" % sim.loss(test_data, {out_p: classification_error}))
        sim.load_params("./model_files/jet_file.ckpt")
        print("parameters loaded")
        print("error after training: %.2f%%" % sim.loss(test_data, {out_p: classification_error}))

    # store trained parameters back into the network
    sim.freeze_params(model)

for conn in model.all_connections:
    conn.synapse = 0.005

if do_training:
    with nengo_dl.Simulator(model, minibatch_size=minibatch_size) as sim:
        print("error w/ synapse: %.2f%%" % sim.loss(test_data, {out_p: classification_error}))

n_presentations = 50
with nengo_loihi.Simulator(model, dt=dt, precompute=False) as sim:
    # if running on Loihi, increase the max input spikes per step
    if 'loihi' in sim.sims:
        sim.sims['loihi'].snip_max_spikes_per_step = 120
    #
    print("Layer 1 number of spikes: ", (sim.data[layer1_p] > 0).sum(axis=0))
    print("Layer 1: ", (sim.data[layer1_p]))
    print("Layer 2 number of spikes: ", (sim.data[layer2_p] > 0).sum(axis=0))
    print("Layer 2: ", (sim.data[layer2_p]))

    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)

    # check classification error
    step = int(presentation_time / dt)
    output = sim.data[out_p][step - 1::step]

    correct = 100 * (np.mean(np.argmax(output, axis=-1) !=
                             np.argmax(test_data[out_p][:n_presentations, -1], axis=-1)))

    print("Predicted labels: ", np.argmax(output, axis=-1))
    print("Correct labels: ", np.argmax(test_data[out_p][:n_presentations, -1], axis=-1))
    print("loihi error: %.2f%%" % correct)
