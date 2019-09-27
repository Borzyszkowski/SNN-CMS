# SNN-CMS
Neuromorphic computing (Spiking Neural Networks) with utilization of Loihi chips in LHC experiments, CMS detector, the European Organization for Nuclear Research.

## Overview

Spiking neural networks are an interesting candidate for signal processing at the High-Luminosity LHC, the next stage of the LHC upgrade. For HL-LHC, new particle detectors will be built, what will allow to take a time-sequence of snapshots for a given collision. This additional information will allow to separate the signal belonging to the interesting collision from those generated parasitic collisions occurring at the same time (in-time pileup) or before/after the interesting one (out-of-time pileup). By powering the LHC real-time processing with spiking neural networks, one could be able to apply advance and accurate signal-to-noise discrimination algorithms in real time, without affecting the overall system latency beyond the given tolerance. 

This project is investigating the potential of Spiking neural networks deployed on neuromorphic chips as a technological solution to increase the precision of the upgraded CMS detector for HL-LHC. We propose to focus on the characterization of a particle type (classification) based on the recorded time profile of the signal, and to determine the arrival time of the particle on the detector (regression). These informations can be used to determine if a particle belongs to the interesting collision or to one of the parasitic collisions. 


### How to run?
>~~~~
>pip install -r requirements.txt
>python prepare_model.py
>snntoolbox conversion_config.txt -t
>~~~~



### Usefull links

* [hls4ML](https://hls-fpga-machine-learning.github.io/hls4ml/) - Firmware implementations of machine learning algorithms using high level synthesis language (HLS)
* [SNNtoolbox](https://snntoolbox.readthedocs.io/en/latest/index.html) - Conversion toolbox containing functions to transform rate-based artificial neural networks into spiking neural networks, and to simulate them
* [Loihi chip overview](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8259423) - Loihi: A Neuromorphic
Manycore Processor with
On-Chip Learning
* [Frontiers in neuroscience - article](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full) - Deep Learning With Spiking Neurons: Opportunities and Challenges
