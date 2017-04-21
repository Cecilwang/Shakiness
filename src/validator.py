# Author: Cecil Wang (cecilwang@126.com)

import numpy as np
from scipy.stats import spearmanr
import utilities

class Validator(object):

    model_proxy = None
    dataset = None
    svr = None

    def __init__(self, model_proxy, dataset, svr):
        self.model_proxy = model_proxy
        self.dataset = dataset
        self.svr = svr

    def SROCC(self, x, y):
        return spearmanr(x, y)[0]

    def accuracy(self, x, y, delta):
        sub = np.absolute(x-y)
        return np.mean(np.absolute(x-y)<delta)

    def cal_score(self, scores):
        return np.mean(scores)

    def validate(self, tool, set, delta=0.5):
        videos = self.dataset.video_queues[set].videos
        x = []
        y = []
        for video in videos:
            print(video[0])
            data, _ = self.dataset.load_samples_from_video(video, cut=False)
            if tool == 'model':
                scores = self.model_proxy.model.predict(data, batch_size=data.shape[0], verbose=1)
            if tool == 'svr':
                scores = self.svr.predict(self.model_proxy, data)
            x.append(self.cal_score(scores))
            #utilities.draw.draw(
            #    [scores,
            #     np.full((scores.shape[0]), video[1]),
            #     np.full((scores.shape[0]), self.cal_score(scores)),]
            #)
            y.append(video[1])
        x = np.array(x)
        y = np.array(y)
        utilities.draw.draw([x,y])

        print('SROCC : ' + str(self.SROCC(x, y)))
        print('ACC : ' + str(self.accuracy(x, y, delta)))
