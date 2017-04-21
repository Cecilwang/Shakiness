# Author: Cecil Wang (cecilwang@126.com)

import keras.backend as K
from keras.metrics import binary_accuracy
from keras.models import load_model
import numpy as np

from model import C3D

class ModelProxy(object):

    name = None
    model = None
    input_shape = None
    features = None

    def my_acc(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.round(y_pred)))

    def __init__(self, name, input_shape, saved_model=None):
        self.name = name
        self.input_shape = input_shape

        if saved_model is not None:
            print("Loading model %s." % saved_model)
            self.model = load_model(saved_model)
        elif name == 'C3D':
            print("Create C3D model.")
            self.model = C3D().create(input_shape)

        input = self.model.input
        output = self.model.layers[20].output
        self.features = K.function([input]+ [K.learning_phase()], [output])

        self.model.compile(loss='mse', optimizer='adadelta',
                           metrics=['accuracy'])

        self.model.summary()

        print('Model has been compiled.')
