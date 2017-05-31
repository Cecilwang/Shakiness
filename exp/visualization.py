# Author: Cecil Wang (cecilwang@126.com)


import numpy as np
from matplotlib import pyplot as plt

from vis.utils import utils
from vis.visualization import visualize_activation

from settings import Settings
from data import Dataset
from model import ModelProxy


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
    )

    layer_name = 'activation_5'

    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model_proxy.model.layers) if layer.name == layer_name][0]

    images = []
    img = visualize_activation(model, layer_idx, max_iter=500)
    images.append(img)

    # Easily stitch images via `utils.stitch_images`
    stitched = utils.stitch_images(images)
    if show:
        plt.axis('off')
        plt.imshow(stitched)
        plt.title('Random imagenet categories')
        plt.show()
