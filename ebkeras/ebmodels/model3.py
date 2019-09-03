# -*- coding: utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense

from .basicmodel import BasicModel


class Model3(BasicModel):

    def __init__(self, *args, **kwargs):
        super(Model3, self).__init__(*args, **kwargs)

        self._model.add(Conv2D(32, (3, 3), input_shape=self._input_shape))
        self._model.add(Activation('relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(64, (3, 3)))
        self._model.add(Activation('relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(128, (3, 3)))
        self._model.add(Activation('relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Flatten())
        self._model.add(Dense(64))
        self._model.add(Activation('relu'))
        self._model.add(Dense(1))
        self._model.add(Activation('sigmoid'))

        self._model.compile(loss='binary_crossentropy',
                            optimizer='rmsprop',
                            metrics=['accuracy'])
