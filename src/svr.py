# Author: Cecil Wang (cecilwang@126.com)

import numpy as np
import pickle
from sklearn.svm import SVR


class MySVR(object):

    svr = None
    dim = None

    def __init__(self, dim, kernel='rbf', saved_svr=None):
        if saved_svr != None:
            self.svr = pickle.load(open(saved_svr, 'rb'))
        else:
            self.svr = SVR(kernel=kernel)
        self.dim = dim

    def feature_aggregation(self, features):
        assert features.shape[1] == self.dim
        mean = np.mean(features, axis=0)
        assert mean.shape[0] == self.dim
        std = np.std(features, axis=0)
        assert std.shape[0] == self.dim

        distribution = np.concatenate((mean, std))
        return distribution.reshape(1, self.dim*2)

    def fit(self, model_proxy, dataset):
        x = np.array([]).reshape(0, self.dim*2)
        y = np.array([])
        videos = dataset.video_queues['train'].videos
        for video in videos:
            print(video[0])
            data, scores = dataset.load_samples_from_video(video, cut=False)
            features = np.array([]).reshape(0, self.dim)
            while data.shape[0] > 0:
                tail = min(data.shape[0], 20)
                data_t = data[:tail, :, :, :]
                data = data[tail:data.shape[0], :, :, :]
                features = np.vstack([model_proxy.features([data_t, 0])[0], features])
            features = self.feature_aggregation(features)
            x = np.vstack([x, features])
            y = np.concatenate([y, scores[:features.shape[0]]])
        self.svr.fit(x, y)

    def predict(self, model_proxy, data):
        features = model_proxy.features([data, 0])[0]
        features = self.feature_aggregation(features)
        return self.svr.predict(features)

    def save(self, filepath):
        pickle.dump(self.svr, open(filepath, 'wb'))
