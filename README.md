# SNN-CMS
Simulation or deployment of Spiking Neural Network for inference on the Loihi chip to solve a Jet Tagging Problem with use of Nengo and NxSDK frameworks.

### How to run?

### Simulation
>~~~~
>pip install -r requirements.txt
>python Jet_SNN_model.py
>~~~~

### On-chip inference
To execute code on the remote Loihi Superhost please configure your machine using following [instruction]( https://www.nengo.ai/nengo-loihi/installation.html). After succesfull installation run the code with a following command:
>~~~~
>SLURM python3 Jet_SNN_model.py
>~~~~

### Usefull links

* [hls4ML](https://hls-fpga-machine-learning.github.io/hls4ml/) - Firmware implementations of machine learning algorithms using high level synthesis language (HLS)
* [Nengo Loihi](https://www.nengo.ai/nengo-loihi/overview.html) - Nengo Loihi is a Python package for running Nengo models on Loihi boards. It contains a Loihi emulator backend for rapid model development and easier debugging, and a Loihi hardware backend for running models on a Loihi board.
* [Loihi chip overview](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8259423) - Loihi: A Neuromorphic
Manycore Processor with
On-Chip Learning
* [Frontiers in neuroscience - article](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full) - Deep Learning With Spiking Neurons: Opportunities and Challenges
