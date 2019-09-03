import os 
import sys

import json

from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


from ebdatagen.basicgen import BasicGenerator

def main(argv, argc):
    
    testing_dir = '../data/new_testing/'
    file_weighst = '/home/ebenezerpdi/Jelambrar/Git/KerasCodes/ebkeras/output_2019-09-03_15-06-48/model_save2019-09-03_15-06-48.txt'
    json_model_file = '/home/ebenezerpdi/Jelambrar/Git/KerasCodes/ebkeras/output_2019-09-03_15-06-48/model.json'
    # file_weighst = 'https://github.com/jelambrar96/KerasCodes/blob/master/ebkeras/output_2019-09-03_15-06-48/model_save2019-09-03_15-06-48.txt'

    input_shape = (64, 64, 1)
    # batch_size = 32

    # basic_gen = BasicGenerator(
    #     input_shape=input_shape,
    #     batch_size=batch_size
    # )

    json_file = open(json_model_file)
    json_model = json_file.read()
    json_file.close()
    
    model = model_from_json(json_model)
    model.load_weights(file_weighst)
    model.compile(loss='binary_crossentropy',
                            optimizer=RMSprop(lr=0.0002),
                            metrics=['accuracy'])


    test_data_gen = ImageDataGenerator(
        rescale=1./255
    )
    test_generator = test_data_gen.flow_from_directory(
        testing_dir,
        target_size=input_shape[:2],
        batch_size=1,
        class_mode='binary', 
        shuffle=False,
        color_mode='grayscale'
    )

    n_files = len(os.listdir(testing_dir))
    probabilities = model.predict(test_generator)

    for index, prob in enumerate(probabilities):
        print('--------------------------------------------------')
        print(test_generator.filenames[index])
        print('prob0: ', prob[0] * 100)
        print('prob0: ', (1.0 - prob[0]) * 100)


if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
