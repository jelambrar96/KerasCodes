
from keras.preprocessing.image import ImageDataGenerator


class DataGenerator:

    def __init__(self, *args, **kwargs):
        
        self._training_dir = kwargs['training_dir']
        self._validation_dir = kwargs['validation_dir']
        self._test_dir = kwargs['testing_dir']

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
        self._training_generator = self._trainig_data_gen.flow_from_directory(
            self._training_dir,
            target_size=self._image_size[:2],
            batch_size=self._batch_size,
            class_mode='binary',
            color_mode=self._color_mode
        )

        self._validation_data_gen = ImageDataGenerator(
            rescale=1./255
        )
        self._validation_generator = self._validation_data_gen.flow_from_directory(
            self._validation_dir,
            target_size=self._image_size[:2],
            batch_size=self._batch_size,
            class_mode='binary',
            color_mode=self._color_mode
        )

        self._test_data_gen = ImageDataGenerator(
            rescale=1./255
        )
        self._test_generator = self._test_data_gen.flow_from_directory(
            self._test_dir,
            target_size=self._image_size[:2],
            batch_size=1,
            class_mode='binary', 
            shuffle=False,
            color_mode=self._color_mode
        )

    def getTrainingGen(self):
        return self._training_generator
    
    def getTestingGen(self):
        return self._test_generator
    
    def getValidation(self):
        return self._validation_generator
   
    