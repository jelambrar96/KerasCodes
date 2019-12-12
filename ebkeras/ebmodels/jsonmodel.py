
from keras.models import model_from_json
from keras.optimizers import RMSprop  # , SGD
from keras.optimizers import SGD

from .basicmodel import BasicModel

class ModelJson(BasicModel):

    def __init__(self, *args, **kwargs):
        print('reading json filename')
        super(ModelJson, self).__init__(*args, **kwargs)

        json_filename = ''
        json_filename = kwargs['json_file']
        
        jfile = open(json_filename, 'r')
        jfile_loaded = jfile.read()
        jfile.close()
        

        self._model = model_from_json(jfile_loaded)
        print('compiling json model...')
        self._model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=SGD(lr=self._learning_rate),
                        metrics=['accuracy'])

