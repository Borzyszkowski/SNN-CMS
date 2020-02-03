""" Script to run a pre-trained neural network on Loihi to solve Jet Tagging Task, using SNN toolbox """

from snntoolbox.bin.run import main

if __name__ == "__main__":
    # path to pre-defined config file
    config_filepath = 'conversion_config_loihi.txt'
    main(config_filepath)
