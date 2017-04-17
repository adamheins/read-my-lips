#!/usr/bin/env python
from __future__ import print_function

import os
import cv2
import dlib
import numpy as np

from keras.preprocessing import image
from keras.layers import Input
from keras_vggface.vggface import VGGFace

import matplotlib.pyplot as plt

# LipNet paper specifies 100 x 50 regions centered on the mouth.
MOUTH_REGION_WIDTH = 100
MOUTH_REGION_HEIGHT = 50


VIDEO_DIRECTORY_PATH = 'test/videos'
ALIGN_DIRECTORY_PATH = 'test/align'
FEATURE_DIRECTORY_PATH = 'data/features'


# Parse alignment file into list of (start, end, word) tuples.
def parse_alignment(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()

    segments = []
    for line in lines:
        words = line.split()

        # Round the start frame down and end frame up.
        start = int(round(float(words[0]) / 1000))
        end = int(round(float(words[1]) / 1000))

        segments.append((start, end, words[2]))

    num_frames = segments[-1][1] - segments[0][0]
    # exclude first and last (cause it's sil)
    return segments[1:-1], num_frames


def process_video(vid_name):
    cap = cv2.VideoCapture(VIDEO_DIRECTORY_PATH + '/' + vid_name)

    align_name = vid_name.split('.')[0]
    alignments, num_frames = parse_alignment(ALIGN_DIRECTORY_PATH + '/' + align_name + '.align')

    frames = np.ndarray(shape=(num_frames, 224, 224, 3), dtype=np.float32)

    frame = 0
    ret, img = cap.read()

    while ret:
        x = cv2.resize(img, (224, 224)).astype(np.float32)

        x = np.expand_dims(x, axis=0)
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 93.5940
        x[:, :, :, 1] -= 104.7624
        x[:, :, :, 2] -= 129.1863

        frames[frame,:,:,:] = x

        ret, img = cap.read()

        frame += 1

    # Divide up frames based on mapping to spoken words.
    word_frames = []
    output = []
    for seg in alignments:
        word_frames.append(frames[seg[0]:seg[1],:,:])
        output.append(seg[2])

    return word_frames, output


def load_data(num_data):
    X = []
    Y = []

    # Get the names of the number of videos we want.
    videos = os.listdir(VIDEO_DIRECTORY_PATH)[0:num_data]

    for vid_name in videos:
        features, output = process_video(vid_name)

        X.extend(features)
        Y.extend(output)

    # Build mapping of vocab to number.
    vocab = {}
    for i, word in enumerate(set(Y)):
        vocab[word] = i

    # Map output to numbers.
    for i, word in enumerate(Y):
        Y[i] = vocab[word]

    # 70% training data
    train_idx = int(len(X)*0.7)
    X_train = X[:train_idx]
    X_test = X[train_idx:]
    Y_train = Y[:train_idx]
    Y_test = Y[train_idx:]

    return (X_train, Y_train), (X_test, Y_test), vocab

