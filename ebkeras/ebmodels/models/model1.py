# -*- coding: utf-8 -*-
# mpodel1.py

from basicmodel import BasicModel
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


class Model1(BasicModel):

    def __init__(self, *args, **kwargs):
        super(BasicModel, self).__init__(args, kwargs)

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
                            optimizer="rmsprop",
                            metrics=["accuracy"])
