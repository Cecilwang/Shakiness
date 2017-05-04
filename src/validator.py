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
        print('Validating.')
        videos = self.dataset.video_queues[set].videos
        videos = sorted(videos, key=lambda x:x[1])
        x = []
        y = []
        index = 0
        total = len(videos)
        for video in videos:
            data, _, _ = self.dataset.load_samples_from_video(video, balance=False)
            if tool == 'model':
                scores = self.model_proxy.model.predict(data, batch_size=data.shape[0], verbose=1)
                scores_a = scores
            if tool == 'svr':
                scores = self.svr.predict(self.model_proxy, data)
            x.append(self.cal_score(scores))
            y.append(video[1])
            index = index+1
            print(str(index)+'/'+str(total))
        x = np.array(x)
        y = np.array(y)
        #utilities.draw.draw([x,y])

        print('SROCC : ' + str(self.SROCC(x, y)))
        print('ACC : ' + str(self.accuracy(x, y, delta)))

        return self.SROCC(x, y)
