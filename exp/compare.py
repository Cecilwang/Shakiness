# Author: Cecil Wang (cecilwang@126.com)

import gc
import matplotlib.pyplot as plt
import seaborn as sns

from keras import backend as K
import numpy as np

from settings import Settings
from data import Dataset
from model import ModelProxy
from svr import MySVR
from validator import Validator
import pickle


if __name__ == '__main__':
    '''
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

    svr = MySVR(Settings().features_dim, saved_svr=Settings().svr['file_path'])



    validator = Validator(model_proxy, dataset, svr)

    _, x1, x2, y = validator.validate('svr', 'test', 5)

    pickle.dump(x1, open('x1', 'wb'))
    pickle.dump(x2, open('x2', 'wb'))
    '''
    dataset = Dataset(
        Settings().nb_samples,
        Settings().videos_description,
        Settings().scores_xlsx,
        Settings().percent_of_test,
        Settings().model_type
    )

    x1 = pickle.load(open('x1', 'rb')).tolist()
    x2 = pickle.load(open('x2', 'rb'))
    m1 = min((x1+x2))
    m2 = max((x1+x2))
    x1 = [ (x-m1)/(m2-m1)*100 for x in x1]
    x2 = [ (x-m1)/(m2-m1)*100 for x in x2]

    y = []
    videos = dataset.video_queues['test'].videos
    for video in videos:
        y.append(video[1])

    sns.set_style("dark")
    plt.plot(x1, label='3D CNN')
    plt.plot(x2, label='SVR')
    plt.plot(y, label='Label')
    plt.legend()
    plt.show()

    sns.distplot(x1, label='3D CNN')
    sns.distplot(x2, label='SVR')
    sns.distplot(y, label='Label')
    plt.legend()
    plt.show()
