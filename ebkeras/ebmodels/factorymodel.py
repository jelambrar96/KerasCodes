# -*- coding: utf-8 -*-

import json

from .model1 import Model1
from .model2 import Model2
from .model3 import Model3
from .model4 import Model4
from .model5 import Model5
from .model6 import Model6


class FactoryModel:

    def getModel(typemode, *args, **kwarg):
        # print(json.dumps(kwarg, indent=2, sort_keys=True))
        typemode = typemode.upper()
        if typemode == 'MODEL1':
            return Model1(*args, **kwarg)
        if typemode == 'MODEL2':
            return Model2(*args, **kwarg)
        if typemode == 'MODEL3':
            return Model3(*args, **kwarg)
        if typemode == 'MODEL4':
            return Model4(*args, **kwarg)
        if typemode == 'MODEL5':
            return Model5(*args, **kwarg)
        if typemode == 'MODEL6':
            return Model6(*args, **kwarg)
        raise Exception('[ERROR] Invalid argument.')
