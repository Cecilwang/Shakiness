# Author: Cecil Wang (cecilwang@126.com)

import numpy as np
import pickle
from sklearn.svm import SVR


class MySVR(object):

    svr = None
    svr_a = None
    dim = None

    def __init__(self, dim, kernel='rbf', saved_svr=None):
        if saved_svr != None:
            self.svr = pickle.load(open(saved_svr, 'rb'))
            self.svr_a = pickle.load(open(saved_svr+'-a', 'rb'))
        else:
            self.svr = SVR(kernel=kernel)
            self.svr_a = SVR(kernel=kernel)
        self.dim = dim

    def feature_aggregation(self, features):
        assert features.shape[1] == self.dim
        mean = np.mean(features, axis=0)
        assert mean.shape[0] == self.dim
        std = np.std(features, axis=0)
        assert std.shape[0] == self.dim
        p50 = np.percentile(features, 50, axis=0)
        assert p50.shape[0] == self.dim
        p75 = np.percentile(features, 75, axis=0)
        assert p75.shape[0] == self.dim
        p100 = np.percentile(features, 100, axis=0)
        assert p100.shape[0] == self.dim

        distribution = np.concatenate((mean, std, p50, p75, p100))
        return distribution.reshape(1, self.dim * 5)
        #return mean.reshape(1, self.dim)

    def fit(self, model_proxy, dataset):
        x = np.array([]).reshape(0, self.dim)
        x_a = np.array([]).reshape(0, self.dim * 5)
        y = np.array([])
        y_a = np.array([])
        w = np.array([])
        w_a = np.array([])
        videos = dataset.video_queues['train'].videos
        for video in videos:
            #print(video[0])
            data, scores, ws = dataset.load_samples_from_video(video, balance=False)
            features = np.array([]).reshape(0, self.dim)
            while data.shape[0] > 0:
                tail = min(data.shape[0], 20)
                data_t = data[:tail, :, :, :]
                data = data[tail:data.shape[0], :, :, :]
                features = np.vstack([model_proxy.features([data_t, 0])[0], features])
            x = np.vstack([x, features])
            y = np.concatenate([y, scores[:features.shape[0]]])
            w = np.concatenate([w, ws[:features.shape[0]]])
            features = self.feature_aggregation(features)
            x_a = np.vstack([x_a, features])
            y_a = np.concatenate([y_a, scores[:features.shape[0]]])
            w_a = np.concatenate([w_a, ws[:features.shape[0]]])
        self.svr.fit(x, y)
        self.svr_a.fit(x_a, y_a)

    def predict(self, model_proxy, data):
        features = model_proxy.features([data, 0])[0]
        y = self.svr.predict(features)
        features = self.feature_aggregation(features)
        y_a = self.svr_a.predict(features)
        return y, y_a

    def save(self, filepath):
        pickle.dump(self.svr, open(filepath, 'wb'))
        pickle.dump(self.svr_a, open(filepath+'-a', 'wb'))
