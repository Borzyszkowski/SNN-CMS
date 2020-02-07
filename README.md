# SNN-CMS
Experiments with different SNN models to solve various problems using SNN Toolbox, Nengo and NxSDK frameworks.

### How to run?
>~~~~
>pip install -r requirements.txt
>~~~~

To execute code on the remote Loihi Superhost please configure your machine using the following [instruction]( https://www.nengo.ai/nengo-loihi/installation.html). After a succesfull installation run the code on Superhost by adding `SLURM=1` to the command.

#### - Spiking autoencoder on MNIST with SNN Toolbox
 `Spiking_autoencoder_MNIST_simulation.py`

 `snntoolbox conversion_config_simulation.txt -t`

#### - Spiking autoencoder on MNIST with Loihi deployment
 `Spiking_autoencoder_MNIST_loihi.py`

#### - Spiking MNIST classifier with Nengo
 `python Nengo_MNIST_Train.py`
 
inspired by https://www.nengo.ai/nengo-loihi/examples/mnist_convnet.html

#### - Optimizing Spiking MNIST with Nengo
 `python Optimizing_SNNs.py`
 
inspired by https://www.nengo.ai/nengo-dl/examples/spiking-mnist.html

#### - Spiking Keyword Spotting Task
 `python Nengo_keyword_spotting.py`
 
inspired by https://www.nengo.ai/nengo-loihi/examples/keyword_spotting.html

#### - Spiking CIFAR 10 classification
 `python Nengo_CIFAR10_conv.py`
 
inspired by https://www.nengo.ai/nengo-extras/examples/cuda_convnet/cifar10_spiking_cnn.html

#### - Spiking Communication Channel
 `python Nengo_communication_channel.py`
 
inspired by https://www.nengo.ai/nengo-loihi/examples/communication_channel.html

#### - Nengo Fashion MNIST
Inserting a Tensorflow / Keras network into the Nengo framework.

`python Nengo_fashion_MNIST.py`
 
inspired by https://www.nengo.ai/nengo-dl/v2.2.0/examples/tensorflow-models.html


### Useful links

* For more information visit the webpage: https://hls-fpga-machine-learning.github.io/hls4ml/
* Presentation about the project at CERN is available [here](https://indico.cern.ch/event/830003/contributions/3523519/?fbclid=IwAR0hQG6KLb1oqnAyZy_GtXAGA23O4FtIIORfAUUhWlLxHRuarscMi1Bmfyc).
* Learn more about [Loihi: A Neuromorphic
Manycore Processor with
On-Chip Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8259423).
* [SNNs overview](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full) - Deep Learning With Spiking Neurons: Opportunities and Challenges
