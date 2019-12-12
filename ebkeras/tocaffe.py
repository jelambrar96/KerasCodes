# -*- coding: utf-8 -*-

import sys
import os

from keras.models import Model
from keras.models import model_from_json

# from ebmodels.keras2caffe import 
from tocaffe import keras2caffe
from tocaffe.convert import convert


def main(argc, argv):
    
    if (argc < 2):
        raise Exception('NO INTPUT FILE JSON')

    json_filename = argv[1]
    if not os.path.isfile(json_filename):
        raise Exception('NO JSON FILE FOUND')

    if (argc < 3):
        raise Exception('NO INTPUT FILE WEIGHTS')
    weights_filename = argv[2]
    if not os.path.isfile(weights_filename):
        raise Exception('NO WEIGHT FILE FOUND')


    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # model = Model()
    model = model_from_json(loaded_model_json)
    
    model.load_weights(weights_filename)
    
    convert(model, 'deploy.txt', 'model.caffemodel')
    


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)