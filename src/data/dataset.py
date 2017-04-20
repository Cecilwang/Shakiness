# Author: Cecil Wang (cecilwang@126.com)

import os
import pickle
import random

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
    videos = []
    video_queues = dict()
    frames = None
    nb_samples = dict()

    image_mode = None

    def __init__(self, nb_videos, videos_description, scores_xlsx,
                 percent_of_test, preprocess=False):
        self.nb_videos = nb_videos
        self.width = videos_description['resize_width']
        self.height = videos_description['resize_height']
        self.clips = videos_description['nb_frames_of_clip']
        self.overlap = videos_description['overlap']
        self.gap = videos_description['gap']
        self.image_mode = IMAGE_MODES[videos_description['image_mode']]

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
                videos_description['frames_path']
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

        print('Loading frames file.')
        self.frames = pickle.load(open(videos_description['frames_path'], 'rb'))
        assert self.nb_videos == len(self.frames)

        self.split_set(percent_of_test)

    def preprocess(self, src_dir, dst_root_dir, frames_file):
        print('Preprocessing videos.')
        filepaths = utilities.file.ls_sorted_dir(src_dir)
        assert self.nb_videos == len(filepaths)
        frames = dict()
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            dst_dir = dst_root_dir + os.path.splitext(filename)[0]
            nb_frames = utilities.video.video2images(
                filepath,
                dst_dir=dst_dir + '/',
                size=(self.width, self.height),
            )
            frames[dst_dir] = nb_frames
        pickle.dump(frames, open(frames_file, 'wb'))

    def split_set(self, percent_of_test=0.0):
        #random.shuffle(self.videos)
        self.nb_test = 0
        self.nb_train = 0

        video_buckets = [[] for i in range(11)]
        for video in self.videos:
            video_buckets[int(video[1]/10)].append(video)

        test_videos = []
        train_videos = []
        for bucket in video_buckets:
            if len(bucket)==0: continue
            nb = max(1, round(len(bucket)*percent_of_test))
            self.nb_test += nb
            self.nb_train += len(bucket)-nb
            test_videos += bucket[:nb]
            train_videos += bucket[nb:]
        assert len(test_videos) == self.nb_test
        assert len(train_videos) == self.nb_train
        assert len(test_videos)+len(train_videos) == len(self.videos)

        self.video_queues['test'] = VideoQueue(test_videos)
        self.nb_samples['test'] = self.get_nb_samples(
            self.video_queues['test'].videos
        )
        random.shuffle(train_videos)
        self.video_queues['train'] = VideoQueue(train_videos)
        self.nb_samples['train'] = self.get_nb_samples(
            self.video_queues['train'].videos
        )

    def get_nb_samples(self, videos):
        nb_samples = 0
        for video in videos:
            nb_frames = self.frames[video[0]]
            nb_frames = len(range(0, nb_frames, 1+self.gap))
            nb_frames = nb_frames - (nb_frames % self.clips) - 1
            stride = int(self.clips - self.clips * self.overlap)
            stride = 1 if stride == 0 else stride
            nb_samples += len(range(0, nb_frames, stride))
        return nb_samples

    def load_samples_from_video(self, video):
        data = utilities.video.load_video_from_images(
            video[0], clips=self.clips, overlap=self.overlap, mode=self.image_mode, gap=self.gap)
        return data, np.full((data.shape[0]), np.round(video[1]), dtype=np.float32)

    def generator(self, video_queue, nb):
        video_queue = self.video_queues[video_queue]
        while True:
            while video_queue.nb_buffer < nb:
                data, scores = self.load_samples_from_video(
                    video_queue.get_video()
                )
                video_queue.append_samples(data, scores)
            x, y = video_queue.get_samples(nb)
            yield (x, y)
