# Author: Cecil Wang (cecilwang@126.com)

from keras import backend as K
from keras.initializers import Constant
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

import utilities


class C3D(metaclass=utilities.singleton.SingletonMetaClass):

    def bn(self, layers, axis=-1):
        return Activation('relu')(
            BatchNormalization(axis=axis)(
                layers
            )
        )

    def conv_3d(self, filters, ksize, strides, input_shape=None):
        if input_shape == None:
            return Convolution3D(filters, ksize, strides=strides,
                                padding='same', kernel_initializer='normal',)
        else:
            return Convolution3D(filters, ksize, strides=strides,
                                padding='same', kernel_initializer='normal',
                                input_shape=input_shape)

    def conv_3d_bn(self, filters, ksize, strides, input_shape=None):
        return self.bn(self.conv_3d(filters, ksize, strides, input_shape), 4)

    def maxpool_3d(self, d, k):
        return MaxPooling3D(
            pool_size=(d, k, k), strides=(d, k, k), padding='same')

    def fc(self, k):
        return Dense(k, kernel_initializer='normal',)

    def fc_bn(self,k):
        return self.bn(self.fc(k))

    def create(self, input_shape):
        model = Sequential()

        model.add(self.conv_3d(64, 3, 1, input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(self.maxpool_3d(1, 2))

        model.add(self.conv_3d(128, 3, 1))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(self.maxpool_3d(2, 2))

        model.add(self.conv_3d(256, 3, 1))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(self.conv_3d_bn(256, 3, 1))
        model.add(self.maxpool_3d(2, 2))

        model.add(self.conv_3d(256, 3, 1))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(self.conv_3d_bn(512, 3, 1))
        model.add(self.maxpool_3d(2, 2))

        model.add(self.conv_3d(256, 3, 1))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(self.conv_3d_bn(512, 3, 1))
        model.add(self.maxpool_3d(2, 2))

        model.add(Flatten())

        model.add(self.fc(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(self.fc(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        return model
