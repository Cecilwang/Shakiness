# Author: Cecil Wang (cecilwang@126.com)

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import time

class Trainer(object):

    model_proxy = None
    dataset = None
    ckpts_dir = None
    logs_dir = None

    def __init__(self, model_proxy, dataset, ckpts_dir, logs_dir):
        self.model_proxy = model_proxy
        self.dataset = dataset
        self.ckpts_dir = ckpts_dir
        self.logs_dir = logs_dir

    def get_callbacks(self, callbacks_set):
        callbacks = []

        if 'ckpt' in callbacks_set:
            callbacks.append(ModelCheckpoint(
                filepath=self.ckpts_dir + self.model_proxy.name + '/' +\
                    '{epoch:03d}-{loss:.3f}.hdf5',
                monitor='loss',
                verbose=1,
                save_best_only=True
            ))

        if 'tb' in callbacks_set:
            callbacks.append(TensorBoard(
                log_dir=self.logs_dir + self.model_proxy.name + '/tb/'))

        if 'early_stopper' in callbacks_set:
            callbacks.append(EarlyStopping(patience=10))

        if 'csv' in callbacks_set:
            timestamp = time.time()
            callbacks.append(
                CSVLogger(self.logs_dir + self.model_proxy.name + '/csv/' + \
                          'training-' + str(time.time()) + '.log')
            )

        return callbacks

    def train(self, batch_size, nb_epochs, callbacks_set, initial_epoch, val):
        if val==True:
            self.model_proxy.model.fit_generator(
                generator=self.dataset.generator('train', batch_size, balance=True),
                steps_per_epoch=self.dataset.nb_samples['train']//batch_size,
                epochs=nb_epochs,
                verbose=2,
                callbacks=self.get_callbacks(callbacks_set),
                initial_epoch=initial_epoch,
                validation_data=self.dataset.generator('test', batch_size, balance=False),
                validation_steps=self.dataset.nb_samples['test']//batch_size,
            )
        else:
            self.model_proxy.model.fit_generator(
                generator=self.dataset.generator('train', batch_size, balance=True),
                steps_per_epoch=self.dataset.nb_samples['train']//batch_size,
                epochs=nb_epochs,
                verbose=2,
                callbacks=self.get_callbacks(callbacks_set),
                initial_epoch=initial_epoch,
            )
