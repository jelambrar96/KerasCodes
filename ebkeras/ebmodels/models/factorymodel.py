# -*- coding: utf-8 -*-

from model1 import Model1
from model2 import Model2
from model3 import Model3
from model4 import Model4
from model5 import Model5


class FactoryModel:

    def getModel(typemode, *args, **kwarg):
        if typemode == 'Model1':
            return Model1(args, kwarg)
        if typemode == 'Model2':
            return Model2(args, kwarg)
        if typemode == 'Model2':
            return Model3(args, kwarg)
        if typemode == 'Model3':
            return Model4(args, kwarg)
        if typemode == 'Model4':
            return Model4(args, kwarg)
        if typemode == 'Model5':
            return Model5(args, kwarg)
        raise Exception('[ERROR] Invalid argument.')
