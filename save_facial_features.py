#!/usr/bin/env python
from __future__ import print_function

import os
import curses
import cv2
import numpy as np

from keras.preprocessing import image
from keras.layers import Input
from keras_vggface.vggface import VGGFace


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


def curses_init():
    stdscr = curses.initscr()
    stdscr.nodelay(True)
    curses.noecho()
    curses.cbreak()
    return stdscr


def curses_clean_up():
    curses.echo()
    curses.nocbreak()
    curses.endwin()


def main():
    stdscr = curses_init()

    # Create facial recognition model.
    input_tensor = Input(shape=(224, 224, 3))
    vgg_model = VGGFace(input_tensor=input_tensor, include_top=False, pooling='avg')

    # Get list of all videos to process.
    videos = os.listdir(VIDEO_DIRECTORY_PATH)
    num_videos = len(videos)

    stdscr.addstr(0, 0, 'Extracting facial features. Press q to exit.')
    stdscr.refresh()

    try:
        for j, video in enumerate(videos):
            stdscr.addstr(1, 0, 'Processed {}/{} videos.'.format(j, num_videos))
            stdscr.addstr(2, 0, 'Processing {}...'.format(video))
            stdscr.refresh()

            word_frames, output = process_video(video)

            name_no_ext = video.split('.')[0]

            # Process each word's set of frames.
            for i, word_frame in enumerate(word_frames):
                # Check if the user has hit q to exit.
                c = stdscr.getch()
                if c == ord('q'):
                    raise Exception

                feature_file_name = '{}_{}_{}'.format(name_no_ext, i, output[i])
                feature_file_path = FEATURE_DIRECTORY_PATH + '/' + feature_file_name

                # If the file already exists, we don't want to waste time processing it again.
                if os.path.isfile(feature_file_path + '.npy'):
                    continue

                # Classify the frames and save the features to a file.
                features = vgg_model.predict(word_frame)
                np.save(feature_file_path, features)
    except:
        # Expected exceptions are either the generic one raised when a user
        # presses 'q' to exit, or a KeyboardInterrupt if the user gets
        # impatient and presses Ctrl-C.
        pass
    finally:
        curses_clean_up()


if __name__ == '__main__':
    main()

