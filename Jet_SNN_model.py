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

# model for Jet classification
with nengo.Network(label="Jet classification") as model:
    nengo_loihi.add_params(model)
    model.config[nengo.Connection].synapse = None

    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])

    neuron_type = nengo.SpikingRectifiedLinear(amplitude=amplitude)
    # neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.001, amplitude=0.005)
    # neuron_type = nengo.AdaptiveLIF(amplitude=0.005)
    # neuron_type = nengo.Izhikevich()

    inp = nengo.Node(np.zeros(n_inputs), label="in")
    out = nengo.Node(size_in=n_outputs)

    layer_1 = nengo.Ensemble(n_neurons=64, dimensions=1, neuron_type=neuron_type, label="Layer 1")
    model.config[layer_1].on_chip = False
    nengo.Connection(inp, layer_1.neurons, transform=nengo_dl.dists.Glorot())
    p1 = nengo.Probe(layer_1.neurons)

    layer_2 = nengo.Ensemble(n_neurons=32, dimensions=1, neuron_type=neuron_type, label="Layer 2")
    nengo.Connection(layer_1.neurons, layer_2.neurons, transform=nengo_dl.dists.Glorot())
    p2 = nengo.Probe(layer_2.neurons)

    layer_3 = nengo.Ensemble(n_neurons=32, dimensions=1, neuron_type=neuron_type, label="Layer 3")
    nengo.Connection(layer_2.neurons, layer_3.neurons, transform=nengo_dl.dists.Glorot())
    p3 = nengo.Probe(layer_3.neurons)

    nengo.Connection(layer_3.neurons, out, transform=nengo_dl.dists.Glorot())

    out_p = nengo.Probe(out)
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01))


def crossentropy(outputs, targets):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets))


def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1), tf.argmax(targets[:, -1], axis=-1)), tf.float32))


dt = 0.001  # simulation timestep
presentation_time = 0.1  # input presentation time
train_data = {inp: train_d[0][:, None, :], out_p: train_d[1][:, None, :]}

# for the test data evaluation we'll be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
test_data = {inp: np.tile(test_d[0][:, None, :], (1, int(presentation_time / dt), 1)),
             out_p_filt: np.tile(test_d[1][:, None, :], (1, int(presentation_time / dt), 1))}

minibatch_size = 200
do_training = True
with nengo_dl.Simulator(model, minibatch_size=minibatch_size, seed=0) as sim:
    if do_training:
        print("error before training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
        # run training
        sim.train(train_data, tf.train.RMSPropOptimizer(learning_rate=0.001),
                  objective={out_p: crossentropy}, n_epochs=20)
        print("error after training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
        sim.save_params("./jet_params")
    else:
        print("error before training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
        sim.load_params("./model_files/jet_file.ckpt")
        print("parameters loaded")
        print("error after training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))

    # store trained parameters back into the network
    sim.freeze_params(model)

for conn in model.all_connections:
    conn.synapse = 0.005

if do_training:
    with nengo_dl.Simulator(model, minibatch_size=minibatch_size) as sim:
        print("error w/ synapse: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))

n_presentations = 50
with nengo_loihi.Simulator(model, dt=dt, precompute=False) as sim:
    # if running on Loihi, increase the max input spikes per step
    if 'loihi' in sim.sims:
        sim.sims['loihi'].snip_max_spikes_per_step = 120

    print("Layer 1 number of spikes: ", (sim.data[p1] > 0).sum(axis=0))
    print("Layer 1: ", (sim.data[p1]))
    print("Layer 2 number of spikes: ", (sim.data[p2] > 0).sum(axis=0))
    print("Layer 2: ", (sim.data[p2]))
    print("Layer 3 number of spikes: ", (sim.data[p3] > 0).sum(axis=0))
    print("Layer 3: ", (sim.data[p3]))

    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)

    # check classification error
    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1::step]

    correct = 100 * (np.mean(np.argmax(output, axis=-1) !=
                             np.argmax(test_data[out_p_filt][:n_presentations, -1], axis=-1)))

    print("Predicted labels: ", np.argmax(output, axis=-1))
    print("Correct labels: ", np.argmax(test_data[out_p_filt][:n_presentations, -1], axis=-1))
    print("loihi error: %.2f%%" % correct)
