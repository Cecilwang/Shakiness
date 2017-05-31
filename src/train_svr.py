# Author: Cecil Wang (cecilwang@126.com)

import tensorflow as tf

from settings import Settings
from data import Dataset
from model import ModelProxy
from svr import MySVR
from validator import Validator


if __name__ == '__main__':
    model_proxy = ModelProxy(
        Settings().model,
        (
            Settings().videos_description['nb_frames_of_clip'],
            Settings().videos_description['crop_height'],
            Settings().videos_description['crop_width'],
            3 if Settings().videos_description['image_mode'] == 'COLOR' else 1,
        ),
        saved_model=Settings().models_dir + Settings().model + Settings().saved_model,
        model_type=Settings().model_type
    )

    dataset = Dataset(
        Settings().nb_samples,
        Settings().videos_description,
        Settings().scores_xlsx,
        Settings().percent_of_test,
        Settings().model_type
        #True
    )

    svr = MySVR(Settings().features_dim)
    svr.fit(model_proxy, dataset)

    validator = Validator(model_proxy, dataset, svr)

    validator.validate('svr', 'test', 5)
