# Author: Cecil Wang (cecilwang@126.com)

import os

import tensorflow as tf

from settings import Settings
from data import Dataset
from model import ModelProxy
from svr import MySVR
from trainer import Trainer
from validator import Validator

def run(model, dataset):
    model_proxy = ModelProxy(
        Settings().model,
        (
            Settings().videos_description['nb_frames_of_clip'],
            Settings().videos_description['crop_height'],
            Settings().videos_description['crop_width'],
            3 if Settings().videos_description['image_mode'] == 'COLOR' else 1,
        ),
        saved_model=Settings().models_dir + Settings().model + '/' + model,
        model_type=Settings().model_type
    )

    svr = MySVR(Settings().features_dim)
    svr.fit(model_proxy, dataset)
    svr.save(Settings().svr['file_path']+'_'+model)

    print(model)
    validator = Validator(model_proxy, dataset, svr)
    return max(validator.validate('svr', 'test', 5))



if __name__ == '__main__':
    dataset = Dataset(
        Settings().nb_samples,
        Settings().videos_description,
        Settings().scores_xlsx,
        Settings().percent_of_test,
        Settings().model_type
        #True
    )

    models = os.listdir(Settings().models_dir + Settings().model)
    max_val = 0
    for model in models:
        s = run(model, dataset)
        if max_val < s:
            max_val = s
            name = model
    print(name)
    print(max_val)
