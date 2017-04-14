#!/usr/bin/env python

import cv2
import dlib
import code
import numpy as np


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

    return segments


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


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_landmarks.dat')

    cap = cv2.VideoCapture('id2_vcd_swwp2s.mpg')
    alignments = parse_alignment('swwp2s.align')
    frames = np.ndarray(shape=(75, MOUTH_REGION_HEIGHT, MOUTH_REGION_WIDTH), dtype=np.uint8)

    frame = 0
    ret, img = cap.read()

    # Extract mouth region from each frame.
    while ret:
        d = detector(img)[0]
        shape = predictor(img, d)

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mouth = get_mouth_rect(grey, shape, MOUTH_REGION_WIDTH, MOUTH_REGION_HEIGHT)
        frames[frame,:,:] = mouth[:,:]

        ret, img = cap.read()

        frame += 1

    cv2.namedWindow('frame')

    # Display deltas.
    key = 0
    for i in xrange(1, frame):
        # Calculate delta frame. Data type needs to be converted to int16 for
        # the subtraction, to avoid problems with overflow and negatives.
        delta = frames[i,:,:].astype('int16') - frames[i-1,:,:].astype('int16')
        delta = np.absolute(delta).astype('uint8')

        cv2.imshow('frame', delta)
        key = cv2.waitKey(40)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
