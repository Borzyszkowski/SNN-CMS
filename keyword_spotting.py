import os
import pickle
import string

import nengo
import numpy as np
try:
    import requests
    has_requests = True
except ImportError:
    has_requests = False

import nengo_loihi


def merge(chars):
    """Merge repeated characters and strip blank CTC symbol"""
    acc = ["-"]
    for c in chars:
        if c != acc[-1]:
            acc.append(c)

    acc = [c for c in acc if c != "-"]
    return "".join(acc)


class Linear(nengo.neurons.NeuronType):
    probeable = ("rates",)

    def __init__(self, amplitude=1):
        super(Linear, self).__init__()

        self.amplitude = amplitude

    def gain_bias(self, max_rates, intercepts):
        """Determine gain and bias by shifting and scaling the lines."""
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)
        gain = max_rates / (1 - intercepts)
        bias = -intercepts * gain
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        intercepts = -bias / gain
        max_rates = gain * (1 - intercepts)
        return max_rates, intercepts

    def step_math(self, dt, J, output):
        """Implement the rectification nonlinearity."""
        output[...] = self.amplitude * J


def download(fname, drive_id):
    """Download a file from Google Drive.

    Adapted from https://stackoverflow.com/a/39225039/1306923
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': drive_id}, stream=True)
    token = get_confirm_token(response)
    if token is not None:
        params = {'id': drive_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, fname)


def load(fname, drive_id):
    if not os.path.exists(fname):
        if has_requests:
            print("Downloading %s..." % fname)
            download(fname, drive_id)
            print("Saved %s to %s" % (fname, os.getcwd()))
        else:
            link = "https://drive.google.com/open?id=%s" % drive_id
            raise RuntimeError(
                "Cannot find '%s'. Download the file from\n  %s\n"
                "and place it in %s." % (fname, link, os.getcwd()))
    print("Loading %s" % fname)
    with open(fname, "rb") as fp:
        ret = pickle.load(fp)
    return ret


n_inputs = 390
n_outputs = 29
n_neurons = 256

allowed_text = ["loha", "alha", "aloa", "aloh", "aoha", "aloha"]
id_to_char = np.array([x for x in string.ascii_lowercase + "\" -|"])

params = load("reference_params.pkl", "149rLqXnJqZPBiqvWpOAysGyq4fvunlnM")
test_stream = load("test_stream.pkl", "1AQavHjQKNu1sso0jqYhWj6zUBLKuGNvV")

# core speech model for keyword spotting
with nengo.Network(label="Keyword spotting") as model:
    nengo_loihi.add_params(model)
    model.config[nengo.Connection].synapse = None

    # network was trained with amplitude of 0.002
    # scaling up improves performance on Loihi
    neuron_type = nengo.LIF(tau_rc=0.02,
                            tau_ref=0.001,
                            amplitude=0.005)

    # below is the core model architecture
    inp = nengo.Node(np.zeros(n_inputs), label="in")

    layer_1 = nengo.Ensemble(n_neurons=n_neurons, dimensions=1,
                             neuron_type=neuron_type,
                             gain=params["x_c_0"]["gain"],
                             bias=params["x_c_0"]["bias"],
                             label="Layer 1")
    model.config[layer_1].on_chip = False
    nengo.Connection(
        inp, layer_1.neurons, transform=params["input_node -> x_c_0"])

    layer_2 = nengo.Ensemble(n_neurons=n_neurons, dimensions=1,
                             neuron_type=neuron_type,
                             gain=params["x_c_1"]["gain"],
                             bias=params["x_c_1"]["bias"],
                             label="Layer 2")
    nengo.Connection(
        layer_1.neurons, layer_2.neurons,
        transform=params["x_c_0 -> x_c_1"])

    char_synapse = nengo.synapses.Alpha(0.005)

    # --- char_out as node
    char_out = nengo.Node(None, label="out", size_in=n_outputs)
    char_output_bias = nengo.Node(params["char_output_bias"])
    nengo.Connection(char_output_bias, char_out, synapse=None)
    nengo.Connection(
        layer_2.neurons, char_out,
        transform=params["x_c_1 -> char_output"])
    char_probe = nengo.Probe(char_out, synapse=char_synapse)

    model.inp = inp


def predict_text(sim, n_steps, p_time):
    """Predict a text transcription from the current simulation state"""
    n_frames = int(n_steps / p_time)
    char_data = sim.data[char_probe]
    n_chars = char_data.shape[1]

    # reshape to separate out each window frame that was presented
    char_out = np.reshape(char_data, (n_frames, p_time, n_chars))

    # take most ofter predicted char over each frame presentation interval
    char_ids = np.argmax(char_out, axis=2)
    char_ids = [np.argmax(np.bincount(i)) for i in char_ids]

    text = merge("".join([id_to_char[i] for i in char_ids]))
    text = merge(text)  # merge repeats to help autocorrect

    return text

stats = {
    "fp": 0,
    "tp": 0,
    "fn": 0,
    "tn": 0,
    "aloha": 0,
    "not-aloha": 0,
}

for arrays, text, speaker_id, _ in test_stream[:10]:
    dt = 0.001
    stream = arrays["inp"]
    assert stream.shape[0] == 1
    stream = stream[0]

    def play_stream(t, stream=stream):
        ti = int(t / dt)
        return stream[ti % len(stream)]

    model.inp.output = play_stream
    n_steps = stream.shape[0]

    sim = nengo_loihi.Simulator(model, dt=dt, precompute=True)
    with sim:
        sim.run_steps(n_steps)

    p_text = predict_text(sim, n_steps, p_time=10)
    print("Predicted:\t%s" % p_text)
    print("Actual:\t\t%s" % text)

    if text == 'aloha':
        stats["aloha"] += 1
        if p_text in allowed_text:
            print("True positive")
            stats["tp"] += 1
        else:
            print("False negative")
            stats["fn"] += 1
    else:
        stats["not-aloha"] += 1
        if p_text in allowed_text:
            print("False positive")
            stats["fp"] += 1
        else:
            print("True negative")
            stats["tn"] += 1
    print("")

print("Summary")
print("=======")
print("True positive rate:\t%.3f" % (stats["tp"] / stats["aloha"]))
print("False negative rate:\t%.3f" % (stats["fn"] / stats["not-aloha"]))
print()
print("True negative rate:\t%.3f" % (stats["tn"] / stats["not-aloha"]))
print("False positive rate:\t%.3f" % (stats["fp"] / stats["aloha"]))

