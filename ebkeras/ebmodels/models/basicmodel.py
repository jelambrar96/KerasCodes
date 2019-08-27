# -*- coding: utf-8 -*-
# basicmodel.py

from keras.models import Sequential


class BasicModel:

    def __init__(self, *args, **kwargs):
        self._model = Sequential()
        self._input_shape = kwargs['input_shape']
        self._epochs = kwargs['epochs']
        self._batch_size = kwargs['batch_size']
        self._class_mode = kwargs['class_mode']
        self._learning_rate = kwargs['learning_rate']

    def load_input_data(self):
        pass

    def train(self):
        pass
