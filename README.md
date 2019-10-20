# SNN-CMS
Simulation or deployment of Spiking Neural Networks for inference on the Loihi chip to solve a Jet Tagging Problem with use of Nengo and NxSDK frameworks.

### How to run?

>~~~~
>pip install -r requirements.txt
>~~~~

To execute code on the remote Loihi Superhost please configure your machine using following [instruction]( https://www.nengo.ai/nengo-loihi/installation.html). After succesfull installation run the code on the Superhost adding `SLURM=1` to the command.

Classification of particles using fully-connected SNNs from the natural representation of Jets with 16 features:
`python Jet_DenseSNN_model.py`

Classification of particles on the images with convolutional SNNs:
`python Jet_ConvSNN_model.py`


### Useful links

* [hls4ML](https://hls-fpga-machine-learning.github.io/hls4ml/) - Firmware implementations of machine learning algorithms using high level synthesis language (HLS)
* [Nengo Loihi](https://www.nengo.ai/nengo-loihi/overview.html) - Nengo Loihi is a Python package for running Nengo models on Loihi boards
* [Loihi chip overview](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8259423) - Loihi: A Neuromorphic
Manycore Processor with
On-Chip Learning
* [Frontiers in neuroscience - article](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full) - Deep Learning With Spiking Neurons: Opportunities and Challenges
