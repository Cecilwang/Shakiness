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
        print('Trainning SVR.')
        x = np.array([]).reshape(0, self.dim * 5)
        y = np.array([])
        w = np.array([])

        print('Extracting features.')
        videos = dataset.video_queues['train'].videos
        index = 0
        total = len(videos)
        for video in videos:
            #print(video[0])
            data, scores, ws = dataset.load_samples_from_video(video, balance=False)
            features = np.array([]).reshape(0, self.dim)
            while data.shape[0] > 0:
                tail = min(data.shape[0], 40)
                data_t = data[:tail, :, :, :]
                data = data[tail:data.shape[0], :, :, :]
                features = np.vstack([model_proxy.features([data_t, 0])[0], features])

            features = self.feature_aggregation(features)
            x = np.vstack([x, features])
            y = np.concatenate([y, scores[:features.shape[0]]])
            w = np.concatenate([w, ws[:features.shape[0]]])
            index = index+1
            print(str(index)+'/'+str(total))

        print('Fitting features.')
        self.svr.fit(x, y)

    def predict(self, model_proxy, data):
        features = np.array([]).reshape(0, self.dim)
        while data.shape[0] > 0:
            tail = min(data.shape[0], 40)
            data_t = data[:tail, :, :, :]
            data = data[tail:data.shape[0], :, :, :]
            features = np.vstack([model_proxy.features([data_t, 0])[0], features])
        features = self.feature_aggregation(features)
        y = self.svr.predict(features)
        return y

    def save(self, filepath):
        pickle.dump(self.svr, open(filepath, 'wb'))
