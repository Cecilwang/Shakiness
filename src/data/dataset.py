# Author: Cecil Wang (cecilwang@126.com)

import os
import pickle
import random

from keras.utils import to_categorical
import numpy as np

import utilities

IMAGE_MODES = dict()
IMAGE_MODES['GRAY'] = 0
IMAGE_MODES['COLOR'] = 1

class VideoQueue(object):

    nb_videos = None
    videos = []
    head = None
    nb_buffer = None
    data_buffer = None
    scores_buffer = None

    def __init__(self, videos):
        self.videos = videos
        self.nb_videos = len(self.videos)
        self.head = 0
        self.nb_buffer = 0

    def get_video(self):
        video = self.videos[self.head]
        self.head = self.head + 1
        self.head = 0 if self.head == self.nb_videos else self.head
        if self.head == len(self.videos):
            self.head = 0
            self.nb_buffer = 0
            self.data_buffer = None
            self.scores_buffer = None
        return video

    def get_samples(self, nb):
        assert self.nb_buffer >= nb
        data = self.data_buffer[:nb, :, :, :, :]
        self.data_buffer = self.data_buffer[nb:, :, :, :, :]
        scores = self.scores_buffer[:nb]
        self.scores_buffer = self.scores_buffer[nb:]
        self.nb_buffer -= nb
        return data, scores

    def append_samples(self, data, scores):
        if self.nb_buffer == 0:
            self.data_buffer = data
            self.scores_buffer = scores
        else:
            self.data_buffer = np.vstack((self.data_buffer, data))
            self.scores_buffer = np.append(self.scores_buffer, scores)
        self.nb_buffer += data.shape[0]

class Dataset(object):

    nb_videos = None
    nb_test = None
    nb_train = None
    width = None
    height = None
    clips = None
    overlap = None
    gap = None
    image_mode = None
    model_type = None
    random_crop = None

    balance = None
    videos = []
    video_queues = dict()
    detail = None
    nb_samples = dict()

    def __init__(self, nb_videos, videos_description, scores_xlsx,
                 percent_of_test, model_type, preprocess=False):
        self.nb_videos = nb_videos
        self.width = videos_description['crop_width']
        self.height = videos_description['crop_height']
        self.clips = videos_description['nb_frames_of_clip']
        self.overlap = videos_description['overlap']
        self.gap = videos_description['gap']
        self.image_mode = IMAGE_MODES[videos_description['image_mode']]
        assert model_type == 'classification' or model_type == 'regression'
        self.model_type = model_type
        self.random_crop = videos_description['random_crop']

        print('Loading scores file.')
        scores = utilities.xlsx.extract_cells(
            scores_xlsx['file_path'],
            scores_xlsx['start_cell'],
            scores_xlsx['end_cell']
        )
        assert self.nb_videos == len(scores)

        if preprocess:
            self.preprocess(
                videos_description['raw_dir'],
                videos_description['processed_dir'],
                videos_description['detail_path']
            )
        else:
            print('Using processed video data.')

        print('Scanning video files.')
        filepaths = utilities.file.ls_sorted_dir(
            videos_description['processed_dir']
        )
        assert self.nb_videos == len(filepaths)

        for i in range(nb_videos):
            self.videos.append((filepaths[i], scores[i]))
        assert self.nb_videos == len(self.videos)

        print('Loading detail file.')
        self.detail = pickle.load(open(videos_description['detail_path'], 'rb'))
        if self.random_crop != 0:
            for k in self.detail.keys():
                self.detail[k]['crops'] = self.random_crop
        assert self.nb_videos == len(self.detail)

        self.split_set(percent_of_test)

        self.balance_train()

        self.nb_samples['test'] = self.get_nb_samples(
            self.video_queues['test'].videos, balance=False
        )

        self.nb_samples['train'] = self.get_nb_samples(
            self.video_queues['train'].videos, balance=True
        )

    def balance_train(self):
        self.balance = [0 for i in range(10)]
        videos = self.video_queues['train'].videos
        for video in videos:
            self.balance[int(video[1]/10)] += self.detail[video[0]]['crops']
        self.balance = [min(10, self.balance[9]/x) for x in self.balance]
        print('balance', self.balance)

    def preprocess(self, src_dir, dst_root_dir, detail_file):
        print('Preprocessing videos.')
        filepaths = utilities.file.ls_sorted_dir(src_dir)
        assert self.nb_videos == len(filepaths)
        detail = dict()
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            dst_dir = dst_root_dir + os.path.splitext(filename)[0]
            nb_frames, nb_crops = utilities.video.video2images(
                filepath,
                dst_dir=dst_dir + '/',
                gray=True,
                crop=[self.height, self.width]
            )
            detail[dst_dir] = {'frames':nb_frames, 'crops':nb_crops}
        pickle.dump(detail, open(detail_file, 'wb'))

    def split_set(self, percent_of_test=0.0):
        #random.shuffle(self.videos)
        self.nb_test = 0
        self.nb_train = 0

        video_buckets = [[] for i in range(10)]
        for video in self.videos:
            video_buckets[int(video[1]/10)].append(video)

        test_videos = []
        train_videos = []
        for bucket in video_buckets:
            if len(bucket)==0: continue
            nb = max(1, round(len(bucket)*percent_of_test))
            nb = min(nb, len(bucket)-1)
            self.nb_test += nb
            self.nb_train += len(bucket)-nb
            test_videos += bucket[:nb]
            train_videos += bucket[nb:]
        assert len(test_videos) == self.nb_test
        assert len(train_videos) == self.nb_train
        assert len(test_videos)+len(train_videos) == len(self.videos)

        self.video_queues['test'] = VideoQueue(test_videos)
        random.shuffle(train_videos)
        self.video_queues['train'] = VideoQueue(train_videos)

    def get_nb_samples(self, videos, balance):
        nb_samples = 0
        tmp = [0,0,0,0,0,0,0,0,0,0]
        tmp_t = [0,0,0,0,0,0,0,0,0,0]
        for video in videos:
            factor = self.balance[int(video[1]/10)]
            nb_frames = self.detail[video[0]]['frames']
            nb_frames = len(range(0, nb_frames, 1+self.gap))
            overlap = self.overlap
            if balance and factor > 1.0:
                overlap = (1.0 - 1.0 / factor)
            stride = round(self.clips - self.clips * overlap)
            stride = 1 if stride == 0 else stride
            nb = 0
            for i in range(0, nb_frames, stride):
                if i + self.clips > nb_frames:
                    break
                nb = nb + 1
            if balance and factor < 1.0:
                nb = round(nb * factor)
            nb *= self.detail[video[0]]['crops']
            nb_samples += nb
            tmp[int(video[1]/10)] += nb
            tmp_t[int(video[1]/10)] += self.detail[video[0]]['crops']
        print("nb videos", tmp_t)
        print("balance nb samples", tmp)
        #utilities.draw.draw([tmp])
        return nb_samples

    def load_samples_from_video(self, video, balance,):
        factor = self.balance[int(video[1]/10)]
        overlap = self.overlap

        # Up sample
        if balance == True and factor > 1.0:
            overlap = (1.0-1.0/factor)

        data = utilities.video.load_video_from_images(
            video[0]+'/',
            self.detail[video[0]]['frames'],
            self.detail[video[0]]['crops'],
            crop=[self.height, self.width],
            mode=self.image_mode,
            clips=self.clips,
            overlap=overlap,
            gap=self.gap,
        )
        assert data.shape[0] == self.detail[video[0]]['crops']

        # Down sample
        if balance == True and factor < 1.0:
            index = np.arange(data.shape[1])
            #np.random.shuffle(index)
            index = index[:round(index.shape[0] * factor)]
            data = data[:, index, :, :, :, :]

        sp = data.shape
        data = data.reshape(sp[0] * sp[1], sp[2], sp[3], sp[4], sp[5])
        scores = np.full((data.shape[0]), video[1], dtype=np.float32)
        weights = np.full((data.shape[0]), factor, dtype=np.float32)
        return data, scores, weights

    def generator(self, video_queue, batch, balance):
        video_queue = self.video_queues[video_queue]
        while True:
            while video_queue.nb_buffer < batch:
                data, scores, _ = self.load_samples_from_video(
                    video_queue.get_video(), balance
                )
                video_queue.append_samples(data, scores)
            x, y = video_queue.get_samples(batch)
            if self.model_type == 'classification':
                y = to_categorical(np.round(y), num_classes=101)
            yield (x, y)
