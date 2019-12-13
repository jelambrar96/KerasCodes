# -*- coding: utf-8 -*-
# mpodel1.py

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD

from .basicmodel import BasicModel


class Model1(BasicModel):

    def __init__(self, *args, **kwargs):
        super(Model1, self).__init__(*args, **kwargs)

        self._input_shape = kwargs['input_shape']

        self._model.add(Conv2D(32, (3, 3), input_shape=self._input_shape))
        self._model.add(Activation("relu"))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(32, (3, 3)))
        self._model.add(Activation("relu"))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Flatten())
        self._model.add(Dense(16))
        self._model.add(Activation("relu"))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(1))
        self._model.add(Activation("sigmoid"))

        self._model.compile(loss="binary_crossentropy",
                            optimizer=SGD(lr=self._learning_rate),
                            metrics=["accuracy"])
