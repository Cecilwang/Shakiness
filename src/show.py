# Author: Cecil Wang (cecilwang@126.com)

from keras.utils import plot_model
import tensorflow as tf

from settings import Settings
from data import Dataset
from model import ModelProxy
from svr import MySVR

import matplotlib.pyplot as plt
import seaborn as sns
import utilities

if __name__ == '__main__':
    dataset = Dataset(
        Settings().nb_samples,
        Settings().videos_description,
        Settings().scores_xlsx,
        Settings().percent_of_test,
        Settings().model_type
        #True
    )

    scores=[]
    for video in dataset.videos:
        scores.append(video[1])
    sns.distplot(scores)
    plt.show()

    tmp = []
    for video in dataset.videos:
        factor = dataset.balance[int(video[1]/10)]
        nb_frames = dataset.detail[video[0]]['frames']
        nb_frames = len(range(0, nb_frames, 1+dataset.gap))
        overlap = dataset.overlap
        if factor > 1.0:
            overlap = (1.0 - 1.0 / factor)
        stride = round(dataset.clips - dataset.clips * overlap)
        stride = 1 if stride == 0 else stride
        nb = 0
        for i in range(0, nb_frames, stride):
            if i + dataset.clips > nb_frames:
                break
            nb = nb + 1
        if factor < 1.0:
            nb = round(nb * factor)
        nb *= dataset.detail[video[0]]['crops']
        for i in range(nb):
            tmp.append(video[1])
    sns.distplot(tmp)
    plt.show()
