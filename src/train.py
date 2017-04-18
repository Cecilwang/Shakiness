# Author: Cecil Wang (cecilwang@126.com)

from settings import Settings
from data import Dataset
#from model import ModelProxy
#from trainer import Trainer


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
        #saved_model=Settings().models_dir + Settings().model + '/003-315.720.hdf5'
    )
    '''

    dataset = Dataset(
        Settings().nb_samples,
        Settings().videos_description,
        Settings().scores_xlsx,
        Settings().percent_of_test,
        True
    )

    '''
    trainer = Trainer(
        model_proxy, dataset, Settings().ckpts_dir, Settings().logs_dir
    )

    trainer.train(
        Settings().batch_size,
        Settings().nb_epochs,
        set(Settings().callbacks_set)
    )
    '''
