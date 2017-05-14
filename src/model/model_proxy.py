# Author: Cecil Wang (cecilwang@126.com)

import keras.backend as K
from keras.initializers import Constant
from keras.layers import Dense
from keras.models import load_model

from model import C3D
from model import preC3D
from utilities.multi_gpu import make_parallel

class ModelProxy(object):

    name = None
    model = None
    input_shape = None
    features = None

    def my_acc(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.round(y_pred)))

    def regression_layer(self):
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.add(Dense(1, use_bias=False, kernel_initializer=(Constant(value=100))))
        self.model.layers[-1].trainable=False

    def regression_compile(self):
        self.model.compile(loss='mse', optimizer='adadelta',
                           metrics=['accuracy'])

    def classification_layer(self):
        self.model.add(Dense(101, activation='softmax'))

    def classification_compile(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])

    def __init__(self, name, input_shape, saved_model=None, model_type='classification'):
        self.name = name
        self.input_shape = input_shape

        if saved_model is not None:
            print("Loading model %s." % saved_model)
            self.model = load_model(saved_model)
        elif name == 'C3D':
            print("Create C3D model.")
            self.model = C3D().create(input_shape)
        elif name == 'preC3D':
            print("Create preC3D model.")
            self.model = preC3D.create(input_shape)

        input = self.model.input
        if name == 'C3D':
            output = self.model.layers[20].output
        if name == 'preC3D':
            output = self.model.layers[29].output
        self.features = K.function([input]+ [K.learning_phase()], [output])

        assert model_type == 'regression' or model_type == 'classification'
        if saved_model is None:
            if model_type == 'regression':
                self.regression_layer()
            if model_type == 'classification':
                self.classification_layer()

        #self.model = make_parallel(self.model, 2)
        
        if model_type=='regression':
            self.regression_compile()
        if model_type=='classification':
            self.classification_compile()

        self.model.summary()

        #print('Model has been compiled.')
