# Author: Cecil Wang (cecilwang@126.com)

from keras.initializers import Constant
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

class C3D():

    def bn(self, layers, axis=-1):
        return Activation('relu')(
            BatchNormalization(axis=axis)(
                layers
            )
        )

    def conv_3d(self, filters, ksize, strides, input_shape=None):
        if input_shape == None:
            return Convolution3D(filters, ksize, strides=strides,
                                 padding='same')
        else:
            return Convolution3D(filters, ksize, strides=strides,
                                 padding='same',
                                 input_shape=input_shape)

    def conv_3d_bn(self, filters, ksize, strides, input_shape=None):
        return self.bn(self.conv_3d(filters, ksize, strides, input_shape), 4)

    def maxpool_3d(self, d, k):
        return MaxPooling3D(
            pool_size=(d, k, k), strides=(d, k, k), padding='same')

    def fc(self, k, name):
        return Dense(k, name=name)

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
        model.add(self.conv_3d(256, 3, 1))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(self.maxpool_3d(2, 2))

        model.add(self.conv_3d(512, 3, 1))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(self.conv_3d(512, 3, 1))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(self.maxpool_3d(2, 2))

        model.add(self.conv_3d(512, 3, 1))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(self.conv_3d(512, 3, 1))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(self.maxpool_3d(2, 2))

        model.add(Flatten())

        model.add(self.fc(4096, 'fc1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(self.fc(4096, 'fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        return model
