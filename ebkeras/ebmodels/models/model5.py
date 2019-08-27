# -*- coding: utf-8 -*-

from basicmodel import BasicModel

from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


class Model5(BasicModel):

    def __init__(self, *args, **kwargs):
        super(BasicModel, self).__init__(args, kwargs)

        self._model.add(Conv2D(32, 3, 3, border_mode='same',
                               input_shape=self._input_shape,
                               activation='relu'))
        self._model.add(Conv2D(32, 3, 3, border_mode='same',
                               activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(64, 3, 3, border_mode='same',
                               activation='relu'))
        self._model.add(Conv2D(64, 3, 3, border_mode='same',
                               activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(128, 3, 3, border_mode='same',
                               activation='relu'))
        self._model.add(Conv2D(128, 3, 3, border_mode='same',
                               activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(256, 3, 3, border_mode='same',
                               activation='relu'))
        self._model.add(Conv2D(256, 3, 3, border_mode='same',
                               activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Flatten())
        self._model.add(Dense(256, activation='relu'))
        self._model.add(Dropout(0.5))

        self._model.add(Dense(256, activation='relu'))
        self._model.add(Dropout(0.5))

        self._model.add(Dense(1))
        self._model.add(Activation('sigmoid'))

        self._model.compile(loss='binary_crossentropy',
                            optimizer=RMSprop(lr=self._learning_rate),
                            metrics=['accuracy'])