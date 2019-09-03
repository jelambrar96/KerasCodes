# -*- coding: utf-8 -*-
import os
import time
import sys

from ebmodels.factorymodel import FactoryModel
from ebdatagen.datagen import DataGenerator




def main(argv, argc):
        
    image_size = 32
    image_height, image_width = (image_size, image_size)
    input_shape = (image_height, image_width)

    MODEL = 'MODEL6'

    str_time = time.strftime('%Y-%m-%d_%H:%M:%S')
    output_folder = 'output_' + str_time
    if os.path.isdir(output_folder):
        print('creating_output_dir')
        os.makedirs(output_folder)
    # data folders

    print(os.path.basename(os.getcwd()))
    print(os.listdir())

    training_dir = '../data/training/'
    if not os.path.isdir(training_dir):
        raise Exception('ERROR: "' + training_dir + '" NO EXISTS')
    
    validation_dir = '../data/validation/'
    if not os.path.isdir(validation_dir):
        raise Exception('ERROR: "' + validation_dir + '" NO EXISTS')
    
    test_dir = '../data/testing/'
    if not os.path.isdir(test_dir):
        raise Exception('ERROR: "' + test_dir + '" NO EXISTS')
    
    epochs = 30
    batch_size = 32
    test_size = 32

    # one channel
    input_shape = (image_height, image_width, 1)

    print('creating model')
    ebmodel = FactoryModel.getModel(
        MODEL,
        input_shape=input_shape,
        epochs=epochs,
        learning_rate=0.001,
        class_mode='binary',
        batch_size=batch_size,
        output_file=os.path.join(output_folder, 'loss_vs_epochs.png')
        )

    print('creating image generator')
    data_generator = DataGenerator(
        training_dir=training_dir,
        validation_dir=validation_dir,
        testing_dir=test_dir,
        input_shape=input_shape,
        batch_size=batch_size
    )

    print('training...')
    ebmodel.train(
        data_generator.getTrainingGen(),
        data_generator.getValidation(),
        os.path.join(output_folder, 'output_log.txt')
    )

    print('saving model')
    
    ebmodel.save_weights(
        os.path.join(output_folder, 'model_save' + str_time + '.txt')
    )

    print('testing..')
    testing_get = data_generator.getTestingGen()
    probabilities = ebmodel.predict(testing_get, 30)

    for index, prob in enumerate(probabilities):
        print('--------------------------------------------------')
        print(testing_get.filenames[index])
        print('prob0: ', prob[0] * 100)
        print('prob0: ', (1 - prob[0]) * 100)

    print('saving summary')
    sumary_string = ebmodel.summary(
        os.path.join(output_folder, 'sumary_model.txt')
    )
    print(sumary_string)

    print('success')



if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
