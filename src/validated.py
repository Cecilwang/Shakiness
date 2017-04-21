# Author: Cecil Wang (cecilwang@126.com)

import tensorflow as tf

from settings import Settings
from data import Dataset
from model import ModelProxy
from validator import Validator


tf.app.flags.DEFINE_string('witch', '', 'Witch use to test?')
FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    model_proxy = ModelProxy(
        Settings().model,
        (
            Settings().videos_description['nb_frames_of_clip'],
            Settings().videos_description['resize_height'],
            Settings().videos_description['resize_width'],
            3 if Settings().videos_description['image_mode'] == 'COLOR' else 1,
        ),
        saved_model=Settings().models_dir + Settings().model + '/790-6.526.hdf5'
    )

    svr = MySVR(saved_svr=Settings().svr['file_path'])

    dataset = Dataset(
        Settings().nb_samples,
        Settings().videos_description,
        Settings().scores_xlsx,
        Settings().percent_of_test,
    )

    validator = Validator(model_proxy, dataset)

    validator.validate(FLAGS.witch, 'test', 5)
    validator.validate(FLAGS.witch, 'train', 5)
