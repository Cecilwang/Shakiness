# Author: Cecil Wang (cecilwang@126.com)

import numpy as np
from scipy.stats import spearmanr

class validator(object):

    model_proxy = None
    dataset = None

    def __init__(self, model_proxy, dataset):
        self.model_proxy = model_proxy
        self.dataset = dataset

    def SROCC(self, x, y):
        return spearmanr(x, y)[0]

    def accuracy(self, x, y, delta):
        return np.mean(np.absolute(x-y)<delta)

    def cal_score(self, scores):
        return np.mean(scores)

    def validate(self, set, delta=0.5):
        videos = self.dataset.video_queues[set].videos
        x = []
        y = []
        for video in videos:
            data, _ = self.dataset.load_samples_from_video(video)
            scores = predict(data, batch_size=data.shape[0], verbose=1)
            x.append(self.cal_score(scores))
            y.append(video[1])
        x = np.array(x)
        y = np.array(y)

        print('SROCC : ' + str(self.SROCC(x, y)))
        print('ACC : ' + str(self.accuracy(x, y, delta)))
