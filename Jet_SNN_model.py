import os
import pickle
import string

import h5py
import glob

import nengo
import numpy as np
import nengo_dl
try:
    import requests
    has_requests = True
except ImportError:
    has_requests = False

import nengo_loihi
import tensorflow as tf

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

n_inputs = 16
n_outputs = 5
n_neurons = 256

# allowed_text = ["loha", "alha", "aloa", "aloh", "aoha", "aloha"]
# id_to_char = np.array([x for x in string.ascii_lowercase + "\" -|"])

# params = load("reference_params.pkl", "149rLqXnJqZPBiqvWpOAysGyq4fvunlnM")
# test_stream = load("test_stream.pkl", "1AQavHjQKNu1sso0jqYhWj6zUBLKuGNvV")


# core speech model for keyword spotting
with nengo.Network(label="Jet classification") as model:
    nengo_loihi.add_params(model)
    model.config[nengo.Connection].synapse = None

    # network was trained with amplitude of 0.002
    # scaling up improves performance on Loihi
    neuron_type = nengo.LIF(tau_rc=0.02,
                            tau_ref=0.001,
                            amplitude=0.005)

    # below is the core model architecture
    inp = nengo.Node(np.zeros(n_inputs), label="in")

    layer_1 = nengo.Ensemble(n_neurons=64, dimensions=1, neuron_type=neuron_type, label="Layer 1")
    model.config[layer_1].on_chip = False
    nengo.Connection(inp, layer_1.neurons, transform=nengo_dl.dists.Glorot())

    layer_2 = nengo.Ensemble(n_neurons=32, dimensions=1, neuron_type=neuron_type, label="Layer 2")
    nengo.Connection(layer_1.neurons, layer_2.neurons, transform=nengo_dl.dists.Glorot())

    layer_3 = nengo.Ensemble(n_neurons=32, dimensions=1, neuron_type=neuron_type, label="Layer 3")
    nengo.Connection(layer_2.neurons, layer_3.neurons, transform=nengo_dl.dists.Glorot())

    char_synapse = nengo.synapses.Alpha(0.005)

    # --- char_out as node
    char_out = nengo.Node(None, label="out", size_in=n_outputs)
    # char_output_bias = nengo.Node(params["char_output_bias"])
    # nengo.Connection(char_output_bias, char_out, synapse=None)
    # nengo.Connection(layer_3.neurons, char_out) #,transform=params["x_c_1 -> char_output"])
    # char_probe = nengo.Probe(char_out, synapse=char_synapse)

    model.inp = inp
    out_p = nengo.Probe(char_out)
    out_p_filt = nengo.Probe(char_out, synapse=nengo.Alpha(0.01))


def crossentropy(outputs, targets):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=outputs, labels=targets))


def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


train_data = features
test_data = target
dt = 0.001  # simulation timestep
presentation_time = 0.1  # input presentation time
max_rate = 100  # neuron firing rates
# neuron spike amplitude (scaled so that the overall output is ~1)
amp = 1 / max_rate
# input image shape
input_shape = 16  ### ????????? ''''''
print(input_shape)
n_parallel = 2  # number of parallel network repetitions
minibatch_size = 200


train_features = features[:60000]
train_targets = target[:60000]
test_features = features[60000:]
test_targets = target[60000:]
test_data = []
train_data = []
train_data.append(train_features)
train_data.append(train_targets)
test_data.append(test_features)
test_data.append(test_targets)


train_data = {inp: train_data[0][:, None, :],
              out_p: train_data[1][:, None, :]}

# for the test data evaluation we'll be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
test_data = {
    inp: np.tile(test_data[0][:, None, :],
                 (1, int(presentation_time / dt), 1)),
    out_p_filt: np.tile(test_data[1][:, None, :],
                        (1, int(presentation_time / dt), 1))
}
do_training = False
with nengo_dl.Simulator(model, minibatch_size=minibatch_size, seed=0) as sim:
    if do_training:
        print("error before training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))

        # run training
        # sim.train(train_data, tf.train.RMSPropOptimizer(learning_rate=0.001),
        #           objective={out_p: crossentropy}, n_epochs=5)

        print("error after training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))

        sim.save_params("./jet_params")
    else:
        # model_files = './model_files'
        # meta_file = glob.glob(model_files + '/*.meta')
        # index_file = glob.glob(model_files + '/*.index')
        # data_file = glob.glob(model_files + '/*.data-00000-of-00001')

        print("error before training: %.2f%%" %
              sim.loss(test_data, {out_p_filt: classification_error}))

        sim.load_params("./model_files/jet_file.ckpt")
        print("parameters loaded")

        print("error after training: %.2f%%" %
              sim.loss(test_data, {out_p_filt: classification_error}))

    # store trained parameters back into the network
    sim.freeze_params(model)

for conn in model.all_connections:
    conn.synapse = 0.005

if do_training:
    with nengo_dl.Simulator(model, minibatch_size=minibatch_size) as sim:
        print("error w/ synapse: %.2f%%" %
              sim.loss(test_data, {out_p_filt: classification_error}))

n_presentations = 50
with nengo_loihi.Simulator(model, dt=dt, precompute=False) as sim:
    # if running on Loihi, increase the max input spikes per step
    if 'loihi' in sim.sims:
        sim.sims['loihi'].snip_max_spikes_per_step = 120

    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)

    # check classification error
    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1::step]
    correct = 100 * (np.mean(
        np.argmax(output, axis=-1)
        != np.argmax(test_data[out_p_filt][:n_presentations, -1],
                     axis=-1)
    ))
    print("loihi error: %.2f%%" % correct)
