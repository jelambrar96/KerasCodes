# -*- coding: utf-8 -*-
# import json

from keras.optimizers import RMSprop  # , SGD
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from .basicmodel import BasicModel


class Model9(BasicModel):

    def __init__(self, *args, **kwargs):

        # print(json.dumps(kwargs, indent=2, sort_keys=True))
        super(Model9, self).__init__(*args, **kwargs)

        self._model.add(
            Conv2D(32, 5, 5, border_mode='same', input_shape=self._input_shape,
                   activation='relu')
            )
        self._model.add(
            Conv2D(32, 3, 3, border_mode='same', activation='relu')
            )
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(
            Conv2D(64, 5, 5, border_mode='same', activation='relu')
            )
        self._model.add(
            Conv2D(64, 3, 3, border_mode='same', activation='relu')
            )
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(
                Conv2D(64, 3, 3, border_mode='same', activation='relu')
            )
        self._model.add(
            Conv2D(64, 3, 3, border_mode='same', activation='relu')
            )
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(
            Conv2D(128, 3, 3, border_mode='same', activation='relu')
            )
        self._model.add(
            Conv2D(128, 3, 3, border_mode='same', activation='relu')
            )
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(
            Conv2D(256, 3, 3, border_mode='same', activation='relu')
            )
        self._model.add(
            Conv2D(256, 3, 3, border_mode='same', activation='relu')
            )
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Flatten())
        self._model.add(Dense(256, activation='relu'))
        self._model.add(Dropout(0.5))

        self._model.add(Dense(256, activation='relu'))
        self._model.add(Dropout(0.5))

        self._model.add(Dense(1))
        self._model.add(Activation('softmax'))

        """
        self._model.compile(loss='binary_crossentropy',
                            optimizer=RMSprop(lr=self._learning_rate),
                            metrics=['accuracy'])
        """
        """
        self._model.compile(loss='binary_crossentropy',
                            optimizer=SGD(lr=self._learning_rate,
                                          decay=0.05*self._learning_rate),
                            metrics=['accuracy'])
        """
        self._model.compile(loss='binary_crossentropy',
                            optimizer=SGD(lr=self._learning_rate),
                            metrics=['accuracy'])
