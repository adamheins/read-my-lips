#!/usr/bin/env python

import os
import cv2
import dlib
import code
import numpy as np

from keras.preprocessing import image
import matplotlib.pyplot as plt

# LipNet paper specifies 100 x 50 regions centered on the mouth.
MOUTH_REGION_WIDTH = 100
MOUTH_REGION_HEIGHT = 50

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


# Returns a rectangle surrounding the mouth region.
def get_mouth_rect(img, shape, w, h):
    # See https://ibug.doc.ic.ac.uk/resources/300-W/ for meaning of different
    # parts.
    points = [shape.part(48), shape.part(51), shape.part(54), shape.part(57)]
    mat = np.matrix([ [p.x, p.y] for p in points ])
    rect = cv2.boundingRect(mat)

    cx = rect[0] + rect[2] / 2
    cy = rect[1] + rect[3] / 2

    top = cy - h / 2
    bottom = cy + h / 2
    left = cx - w / 2
    right = cx + w / 2

    return img[top:bottom, left:right]


def load_data():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_landmarks.dat')

    X = []
    Y = []
    

    for vid_name in os.listdir('test/videos'):
        cap = cv2.VideoCapture('test/videos/'+vid_name)

        align_name = vid_name.split('.')[0]
        alignments, num_frames = parse_alignment('test/align/' + align_name + '.align')

        frames = np.ndarray(shape=(num_frames, 224, 224, 3), dtype=np.float32)

        frame = 0
        ret, img = cap.read()

        # for debugging the first frame
        # return x

        while ret:
            # d = detector(img)[0]
            # shape = predictor(img, d)

            # grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # mouth = get_mouth_rect(grey, shape, MOUTH_REGION_WIDTH, MOUTH_REGION_HEIGHT)

            x = cv2.resize(img, (224, 224)).astype(np.float32)

            x = np.expand_dims(x, axis=0)	
	    # Zero-center by mean pixel
            x[:, :, :, 0] -= 93.5940
            x[:, :, :, 1] -= 104.7624
            x[:, :, :, 2] -= 129.1863

            # frames[frame,:,:] = mouth[:,:]
            frames[frame,:,:,:] = x

            ret, img = cap.read()

            frame += 1

        # cv2.namedWindow('frame')

        # Display deltas.
        # key = 0
        # for i in xrange(1, frame):
        #     # Calculate delta frame. Data type needs to be converted to int16 for
        #     # the subtraction, to avoid problems with overflow and negatives.
        #     delta = frames[i,:,:].astype('int16') - frames[i-1,:,:].astype('int16')
        #     delta = np.absolute(delta).astype('uint8')

        #     cv2.imshow('frame', delta)
        #     key = cv2.waitKey(40)

        # cv2.destroyAllWindows()

        # group frames with corresponding align word
        # (start, end, word)

        for seg in alignments:
            X.append(frames[seg[0]:seg[1],:,:])
            Y.append(seg[2])
    
    # 70% training data
    train_idx = int(len(X)*0.7)
    X_train = X[:train_idx]
    X_test = X[train_idx:]
    Y_train = Y[:train_idx]
    Y_test = Y[train_idx:]
    
    return (X_train, Y_train), (X_test, Y_test)
