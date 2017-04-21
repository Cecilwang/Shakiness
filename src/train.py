# Author: Cecil Wang (cecilwang@126.com)

import tensorflow as tf

from settings import Settings
from data import Dataset
from model import ModelProxy
from svr import MySVR
from trainer import Trainer


tf.app.flags.DEFINE_string('witch', '', 'Witch to train?')
FLAGS = tf.app.flags.FLAGS


def train_model(model_proxy, dataset):
    trainer = Trainer(
        model_proxy, dataset, Settings().ckpts_dir, Settings().logs_dir
    )

    trainer.train(
        Settings().batch_size,
        Settings().nb_epochs,
        set(Settings().callbacks_set),
        0,
        False
    )

def train_svr(model_proxy, dataset):
    svr = MySVR()
    svr.fit(model_proxy, dataset)
    svr.save(Settings().svr['file_path'])


if __name__ == '__main__':
    model_proxy = ModelProxy(
        Settings().model,
        (
            Settings().videos_description['nb_frames_of_clip'],
            Settings().videos_description['resize_height'],
            Settings().videos_description['resize_width'],
            3 if Settings().videos_description['image_mode'] == 'COLOR' else 1,
        ),
        #saved_model=Settings().models_dir + Settings().model + '/669-2.259.hdf5'
    )

    dataset = Dataset(
        Settings().nb_samples,
        Settings().videos_description,
        Settings().scores_xlsx,
        Settings().percent_of_test,
        #True
    )

    if FLAGS.witch == 'svr':
        train_svr(model_proxy, dataset)
    if FLAGS.witch == 'model':
        train_model(model_proxy, dataset)
