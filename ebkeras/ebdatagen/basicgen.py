
from keras.preprocessing.image import ImageDataGenerator


class BasicGenerator:
    
    def __init__(self, *args, **kwargs):
        
        # self._training_dir = kwargs['training_dir']
        # self._validation_dir = kwargs['validation_dir']
        # self._test_dir = kwargs['testing_dir']

        self._image_size = kwargs['input_shape']
        self._batch_size = kwargs['batch_size']

        self._color_mode = 'rgb'
        if len(self._image_size) == 2:
            self._color_mode = 'grayscale'
        elif self._image_size[2] == 1:
            self._color_mode = 'grayscale'
        elif self._image_size[2] == 4:
            self._color_mode = 'rgba'
    
        self._trainig_data_gen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False,
                vertical_flip=False
            )
    
    def getImageGenerator(self):
        return self._trainig_data_gen