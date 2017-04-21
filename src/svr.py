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
        x = np.array([]).reshape(0, 4096)
        y = np.array([])
        videos = dataset.video_queues['train'].videos
        for video in videos:
            print(video[0])
            data, scores = dataset.load_samples_from_video(video, cut=False)
            features = model_proxy.features([data, 0])[0]
            x = np.vstack([x, features])
            y = np.concatenate([y, scores])
        self.svr.fit(x, y)

    def predict(self, model_proxy, data):
        features = model_proxy.features([data, 0])[0]
        return self.svr.predict(features)

    def save(self, filepath):
        pickle.dump(self.svr, open(filepath, 'wb'))
