# SNN-CMS
Simulation of Spiking Neural Networks in SNNToolbox to solve the Jet Tagging Problem.

### How to run?

##### training and simulation
>~~~~
>pip install -r requirements.txt
>python prepare_model.py
>snntoolbox conversion_config.txt -t
>~~~~
##### training and hardware run
>~~~~
>SLURM=1 SNNT_deployment.py
>~~~~

### Useful links

* For more information visit the webpage: https://hls-fpga-machine-learning.github.io/hls4ml/
* Presentation about the project at CERN is available [here](https://indico.cern.ch/event/830003/contributions/3523519/?fbclid=IwAR0hQG6KLb1oqnAyZy_GtXAGA23O4FtIIORfAUUhWlLxHRuarscMi1Bmfyc).
* Learn more about [Loihi: A Neuromorphic
Manycore Processor with
On-Chip Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8259423).
* [SNNs overview](https://www.frontiersin.org/articles/10.3389/fnins.2018.00774/full) - Deep Learning With Spiking Neurons: Opportunities and Challenges
