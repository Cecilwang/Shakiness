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


def clips_video(video_data, clips, overlap):
    nb_frames = len(video_data)
    stride = round(clips - clips * overlap)
    stride = 1 if stride == 0 else stride
    clips_data = []
    for i in range(0, nb_frames, stride):
        if i + clips > nb_frames:
            break
        clips_data.append(video_data[i:i+clips,:,:,:])
    return clips_data


def load_video_from_images(video_dir, nb_frames, nb_crops, crop, mode=0, size=None, clips=None, overlap=0.0, gap=0):
    data = []
    data_t = []
    if size == None:
        frames = [cv2.imread(video_dir+str(i)+'.png', mode) for i in range(nb_frames)]
    else:
        frames = [cv2.resize(cv2.imread(video_dir+str(i)+'.png', mode), size) for i in range(nb_frames)]
    for frame in frames:
        data_t.append(crop_image(np.expand_dims(frame, 2), crop[0], crop[1]))
    data_t = np.transpose(data_t, (1, 0, 2, 3, 4))
    for i in range(data_t.shape[0]):
        data.append(clips_video(data_t[i,:,:,:,:], clips, overlap))
    return np.array(data)


def crop_image(image, h_size, w_size):
    h, w, c = image.shape

    h_nb = h // h_size
    w_nb = w // w_size

    h_boundary = (h % h_size) // 2
    w_boundary = (w % w_size) // 2

    images = []

    for i in range(h_nb):
        for j in range(w_nb):
            y = h_boundary + i * h_size
            x = w_boundary + j * w_size
            images.append(image[y:y+h_size, x:x+w_size, :])

    return images


@ProgressBar(10, 100)
def video2images(src_file, dst_dir, size=None, gray=False, crop=None):
    if os.path.isdir(dst_dir) == False:
        os.mkdir(dst_dir)

    cap = cv2.VideoCapture(src_file)
    assert cap.isOpened()

    nb_crops = 0
    if crop != None:
        h, w = get_video_properties(cap)[1:3]
        nb_crops = (h // crop[0]) * (w // crop[1])

    nb_frames = 0
    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        if size != None:
            frame = cv2.resize(frame, size)
        if gray == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(dst_dir+str(nb_frames)+'.png', frame)
        nb_frames += 1
    cap.release()
    return nb_frames, nb_crops


if __name__ == '__main__':
    pass
