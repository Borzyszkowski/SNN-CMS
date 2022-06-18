<p align="center">

  <h1 align="center">Neuromorphic Computing in High Energy Physics</h1>
  <p align="center">
    <strong>Bartłomiej Borzyszkowski</strong>
    ·
      <strong>Jean-Roch Vlimant</strong>
    ·
      <strong>Maurizio Pierini</strong>

  </p>
  
  <h5 align="center">European Organization for Nuclear Research (CERN)
    <br>
   Espl. des Particules 1, 1211 Meyrin, Switzerland
  </h5>

  
  <div align="center">
  </div>
<br>
<p align="center"><img src="https://openlab.cern/sites/default/files/LOGO_CERN_openlab_0.png" width="300" align="middle"></p>



  <p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href=https://doi.org/10.5281/zenodo.3755310><img alt="Paper" src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'></a>
    
         
   
  </p>
</p>
<br>













In this project, we present novel applications of Spiking Neural Networks with utilization of Loihi chips in LHC experiments, at CMS detector.
The work is developed by the European Organization for Nuclear Research (CERN) in collaboration with Intel Labs.

## Overview

Spiking neural networks are an interesting candidate for signal processing at the High-Luminosity LHC, the next stage of the LHC upgrade. For HL-LHC, new particle detectors will be built, what will allow to take a time-sequence of snapshots for a given collision. This additional information will allow to separate the signal belonging to the interesting collision from those generated parasitic collisions occurring at the same time (in-time pileup) or before/after the interesting one (out-of-time pileup). By powering the LHC real-time processing with spiking neural networks, one could be able to apply advance and accurate signal-to-noise discrimination algorithms in real time, without affecting the overall system latency beyond the given tolerance. 

This project is investigating the potential of Spiking neural networks deployed on neuromorphic chips as a technological solution to increase the precision of the upgraded CMS detector for HL-LHC. We propose to focus on the characterization of a particle type (classification) based on the recorded time profile of the signal, and to determine the arrival time of the particle on the detector (regression). These informations can be used to determine if a particle belongs to the interesting collision or to one of the parasitic events. 

### How to run?
>~~~~
>git clone https://github.com/Borzyszkowski/SNN-CMS.git
>~~~~

##### Jet Tagging - Explore dataset:
>~~~~
>git checkout Jet_Tagging_data
>~~~~

##### Jet Tagging - Simulation and Loihi on-chip inference with SNNToolbox:
>~~~~
>git checkout SNNToolbox_model
>~~~~

##### Jet Tagging - Simulation and Loihi on-chip inference with Nengo and NxSDK:
>~~~~
>git checkout Nengo_model
>~~~~

##### Other neuromorphic experiments with SNN Toolbox, Nengo and NxSDK:
>~~~~
>git checkout Neuromorphic_experiments
>~~~~

Use specific branches of the repository and follow instructions of README file in every branch.

### Useful links

* For more information visit the webpage: https://hls-fpga-machine-learning.github.io/hls4ml/
* Presentation about the project at CERN is available [here](https://indico.cern.ch/event/830003/contributions/3523519/?fbclid=IwAR0hQG6KLb1oqnAyZy_GtXAGA23O4FtIIORfAUUhWlLxHRuarscMi1Bmfyc).
* Learn more about [Loihi: A Neuromorphic
Manycore Processor with
On-Chip Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8259423).
* [SNNs overview](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full) - Deep Learning With Spiking Neurons: Opportunities and Challenges

### Contact
* Email: maurizio.pierini@cern.ch

### Citation
Please cite our work as follows:

```
@misc{bartlomiej_borzyszkowski_2020_3755310,
  author       = {Bartlomiej Borzyszkowski},
  title        = {Neuromorphic Computing in High Energy Physics},
  month        = apr,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3755310},
  url          = {https://doi.org/10.5281/zenodo.3755310}
}
```
