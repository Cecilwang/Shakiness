# Author: Cecil Wang (cecilwang@126.com)

import numpy as np
import pickle
from sklearn.svm import SVR


class MySVR(object):

    svr = None

    def __init__(self, kernel='rbf', saved_svr=None):
        if saved_svr != None:
            self.svr = pickle.load(open(saved_svr, 'rb'))
        else:
            self.svr = SVR(kernel=kernel)

    def fit(self, model_proxy, dataset):
        videos = dataset.video_queues['train'].videos
        for video in videos:
            data, scores = dataset.load_samples_from_video(video, cut=False)
            features = model_proxy.features([data, 0])[0]
            self.svr.fit(features, scores)

    def predict(self, model_proxy, data):
        features = self.model_proxy.features([data, 0])[0]
        self.svr.predict(features)

    def save(self, filepath):
        pickle.dump(self.svr, open(filepath, 'wb'))
