""" Convert model in json and h5 formats into a single h5 file. """

from keras.models import model_from_json


def gen_h5(json_path, h5_path):
    # load json and create model
    with open(json_path) as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print("Model from the disk loaded")

    # load weights into new model
    loaded_model.load_weights(h5_path)
    print("Weights from the disk loaded")

    # save model as h5
    loaded_model.save('model2SNN.h5')
    print("h5 file with model and weights saved")
