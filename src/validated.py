# Author: Cecil Wang (cecilwang@126.com)

from settings import Settings
from data import Dataset
from model import ModelProxy
from trainer import Trainer


if __name__ == '__main__':
    '''
    model_proxy = ModelProxy(
        Settings().model,
        (
            Settings().videos_description['nb_frames_of_clip'],
            Settings().videos_description['resize_height'],
            Settings().videos_description['resize_width'],
            3 if Settings().videos_description['image_mode'] == 'COLOR' else 1,
        ),
        saved_model=Settings().models_dir + Settings().model + '/669-2.259-0.2.hdf5'
    )
    '''

    dataset = Dataset(
        Settings().nb_samples,
        Settings().videos_description,
        Settings().scores_xlsx,
        Settings().percent_of_test,
    )
