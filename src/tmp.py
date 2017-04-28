# Author: Cecil Wang (cecilwang@126.com)

import cv2
import utilities
from data import Dataset
from settings import Settings
import numpy as np

if __name__ == '__main__':
    '''
    utilities.video.video2images('D:/Project/shakiness/data/video/raw/1.webm',
                                 'D:/Project/shakiness/data/tmp/1/',
                                 gray=True,
                                 crop=[200,300])


    videos = utilities.file.ls_sorted_dir('D:/Project/shakiness/data/video/raw/')
    for i in range(len(videos)):
        cap = cv2.VideoCapture(videos[i])
        n,h,w=utilities.video.get_video_properties(cap)
        print(h,w)
        cap.release()

    utilities.video.load_video_from_images('D:/Project/shakiness/data/tmp/1/',
                                            238, 9,
                                            gap=1,
                                            clips=16,
                                            overlap=0.5)
    '''
    dataset = Dataset(
        Settings().nb_samples,
        Settings().videos_description,
        Settings().scores_xlsx,
        Settings().percent_of_test,
        Settings().model_type,
        #True
    )
    print(dataset.nb_samples)

    '''
    import time
    t = time.time()
    nb = 0
    videos = dataset.video_queues['test'].videos
    for video in videos:
        print(video[0])
        data, _, _ = dataset.load_samples_from_video(video, balance=False)
        data.astype(np.float32)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    image = data[i,j,k,:,:,:]
                    print(image.shape)
                    cv2.imshow('haha', image)
                    cv2.waitKey(0)
        nb += data.shape[0]*data.shape[1]
    print(nb)
    print(time.time()-t)
    '''
