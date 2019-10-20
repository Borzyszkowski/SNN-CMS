# SNN-CMS
Simulation and deployment of Spiking Neural Networks for inference on the Loihi chip to solve the Jet Tagging Problem with use of Nengo and NxSDK frameworks.

### How to run?

>~~~~
>pip install -r requirements.txt
>~~~~

To execute code on the remote Loihi Superhost please configure your machine using the following [instruction]( https://www.nengo.ai/nengo-loihi/installation.html). After a succesfull installation run the code on Superhost by adding `SLURM=1` to the command.

Classification of particles using fully-connected SNNs from the natural representation of Jets with 16 features:

`python Jet_DenseSNN_model.py`

Classification of particles on the images with convolutional SNNs:

`python Jet_ConvSNN_model.py`


### Useful links

* For more information visit the webpage: https://hls-fpga-machine-learning.github.io/hls4ml/
* Presentation about the project at CERN is available [here](https://indico.cern.ch/event/830003/contributions/3523519/?fbclid=IwAR0hQG6KLb1oqnAyZy_GtXAGA23O4FtIIORfAUUhWlLxHRuarscMi1Bmfyc).
* Learn more about [Loihi: A Neuromorphic
Manycore Processor with
On-Chip Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8259423).
* [SNNs overview](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full) - Deep Learning With Spiking Neurons: Opportunities and Challenges
