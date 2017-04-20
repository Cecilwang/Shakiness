# Author: Cecil Wang (cecilwang@126.com)

import os

import cv2
import numpy as np

from utilities.decorator import ProgressBar
from utilities.file import ls_sorted_dir


def get_video_properties(cap):
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )


def play_video(filepath):
    cap = cv2.VideoCapture(filepath)
    assert cap.isOpened()

    while(True):
        ret, frame = cap.read()
        if ret == False:
            break;
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_nb_clips(filepath, clips, overlap=0.0):
    cap = cv2.VideoCapture(filepath)
    assert cap.isOpened()
    nb_frames = 0
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        nb_frames += 1
    nb_frames = nb_frames - (nb_frames % clips) - 1
    stride = int(clips - clips * overlap)
    stride = 1 if stride == 0 else stride
    return len(range(0, nb_frames, stride))


def clips_video(video_data, clips, overlap):
    nb_frames = len(video_data)
    nb_frames = nb_frames - (nb_frames % clips) - 1
    stride = int(clips - clips * overlap)
    stride = 1 if stride == 0 else stride
    clips_data = []
    for i in range(0, nb_frames, stride):
        clips_data.append(video_data[i:(i+clips), :, :, :])
    clips_data = np.array(clips_data).astype(np.float32)
    return clips_data


def load_video(filepath, size=None, clips=None, overlap=0.0):
    cap = cv2.VideoCapture(filepath)
    assert cap.isOpened()

    video_data = []
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        if size != None:
            frame = cv2.resize(frame, size)
        video_data.append(frame)
    video_data = np.array(video_data).astype(np.float32)

    if clips != None:
        video_data = clips_video(video_data, clips, overlap)

    cap.release()
    return video_data


def play_video_from_array(data):
    for i in range(data.shape[0]):
        cv2.imshow('frame', data[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def play_video_from_images(video_dir, mode):
    image_files = ls_sorted_dir(video_dir)
    data = [cv2.imread(x, mode) for x in image_files]
    data = np.array(data)
    print(data.shape)
    play_video_from_array(data)


def load_video_from_images(video_dir, size=None, clips=None, overlap=0.0, mode=-1, gap=0):
    image_files = ls_sorted_dir(video_dir)
    image_files = [image_files[i] for i in range(0, len(image_files), 1+gap)]

    if size == None:
        data = [cv2.imread(x, mode) for x in image_files]
    else:
        data = [cv2.resize(cv2.imread(x, mode), size) for x in image_files]
    data = np.array(data).astype(np.float32)
    if mode == 0:
        data = np.expand_dims(data, 3)

    if clips != None:
        data = clips_video(data, clips, overlap)

    return data


@ProgressBar(10, 100)
def video2images(src_file, dst_dir, size=None,):
    if os.path.isdir(dst_dir) == False:
        os.mkdir(dst_dir)

    cap = cv2.VideoCapture(src_file)
    assert cap.isOpened()

    nb_frames = 0
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        nb_frames += 1
        if size != None:
            frame = cv2.resize(frame, size)
        cv2.imwrite(dst_dir+str(nb_frames)+'.png', frame)
    cap.release()
    return nb_frames


if __name__ == '__main__':

    play_video_from_images(
        'D:/Project/Shakiness/data/video/processed/1/', -1
    )
