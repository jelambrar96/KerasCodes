# -*- coding: utf-8 -*-
# basicmodel.py

# import json

from keras.models import Sequential
from keras.models import model_from_json
# from keras.models import save_model
from keras.callbacks import CSVLogger

# from livelossplot import PlotLossesKeras
from .plossess import PlotLosses


class BasicModel:

    def __init__(self, *args, **kwargs):
        # print(json.dumps(kwargs, indent=2, sort_keys=True))
        self._model = Sequential()
        self._input_shape = kwargs['input_shape']
        self._epochs = kwargs['epochs']
        self._batch_size = kwargs['batch_size']
        self._class_mode = kwargs['class_mode']
        self._learning_rate = kwargs['learning_rate']
        self._output_file = kwargs['output_file']
        self._plot_losses = PlotLosses(self._output_file)

    def get_test_data_gen(self):
        pass

    def load_input_data(self):
        pass

    def load_model_from_json(self, jsonfile):
        json_file = open(jsonfile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self._model = model_from_json(loaded_model_json)

    def load_weights(self, weights_file):
        self._model.load_weights(weights_file)

    def load_model(self, json_file, weights_file):
        json_file = open(json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.load_model_from_json(loaded_model_json)
        self._model.load_weights(weights_file)

    def predict(self, test_generator, size=None):
        if size is None:
            return self._model.predict_generator(test_generator)
        return self._model.predict_generator(test_generator, size)

    def test(self):
        pass

    def toJson(self):
        return self._model.to_json()

    def train(self, training_generator, validator_generator, logfile,
              epochs=None, batch_size=None):
        if epochs is None:
            epochs = self._epochs
        if batch_size is None:
            batch_size = self._batch_size
        self._model.fit(
            training_generator,
            steps_per_epoch=len(training_generator.filenames) // batch_size,
            epochs=epochs,
            validation_data=validator_generator,
            validation_steps=len(validator_generator.filenames) // batch_size,
            callbacks=[self._plot_losses,
                       CSVLogger(logfile, separator=';', append=False)],
            verbose=2
        )

    def save_weights(self, filename):
        self._model.save_weights(filename)

    def save_model_json(self, filename):
        with open(filename, 'w') as json_file:
            json_file.write(self._model.to_json())

    def summary(self, filename=''):
        if filename:
            with open(filename, 'w') as fout:
                self._model.summary(
                        print_fn=lambda line: fout.write(line + '\n'))
        return self._model.summary()
