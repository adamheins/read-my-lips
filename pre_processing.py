#!/usr/bin/env python
from __future__ import print_function

import os
import curses
import cv2
import glob
import numpy as np
import sys

from keras.layers import Input
from keras_vggface.vggface import VGGFace


VIDEO_DIRECTORY_PATH = 'data/videos'
ALIGN_DIRECTORY_PATH = 'data/align'
FEATURE_DIRECTORY_PATH = 'data/features'

DEFAULT_TRAINING_FRACTION = 0.7


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


def process_video(speaker, vid_name):
    video_path = os.path.join(VIDEO_DIRECTORY_PATH, speaker, vid_name)
    align_name = vid_name.split('.')[0] + '.align'
    align_path = os.path.join(ALIGN_DIRECTORY_PATH, speaker, align_name)

    cap = cv2.VideoCapture(video_path)
    alignments, num_frames = parse_alignment(align_path)

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


def build_vocab(y):
    # Build mapping of vocab to number.
    vocab = {}
    for i, word in enumerate(set(y)):
        vocab[word] = i

    # Map output to numbers.
    output_vector = []
    for word in y:
        output_vector.append(vocab[word])

    return vocab, output_vector


def split_train_test(x, y, train_fraction):
    ''' Split data into training and testing data sets. '''
    train_idx = int(len(x) * train_fraction)

    x_train = np.asarray(x[:train_idx])
    x_test = np.asarray(x[train_idx:])

    y_train = np.asarray(y[:train_idx])
    y_test = np.asarray(y[train_idx:])

    return (x_train, y_train), (x_test, y_test)


def load_data(num_words=0, train_fraction=DEFAULT_TRAINING_FRACTION, speaker=None):
    ''' Load facial feature data from disk. '''
    # If num_words is greater than 0, only that many word files with be used as
    # input. Otherwise, all available will be used.

    x = [] # Input
    y = [] # Desired output

    # Select data files to load. If a speaker is selected, only their data
    # files will be used. Otherwise, data files with be selected from all
    # speakers.
    if speaker:
        data_glob = os.path.join(FEATURE_DIRECTORY_PATH, speaker, '*.npy')
    else:
        data_glob = os.path.join(FEATURE_DIRECTORY_PATH, '*', '*.npy')

    data_files = glob.glob(data_glob)
    if len(data_files) < num_words:
        data_files = data_files[0:num_words]

    # Load the data.
    for data_file in data_files:
        name_no_ext = os.path.basename(data_file).split('.')[0]
        word = name_no_ext.split('_')[2]

        data = np.load(data_file)

        x.append(data)
        y.append(word)

    # Build mapping of vocabulary to integers, and remap output to it.
    vocab, y = build_vocab(y)

    # Calculate the maximum number of frames any word has.
    word_max_frames = 0
    for word in x:
        if word.shape[0] > word_max_frames:
            word_max_frames = word.shape[0]

    # Pad features up to maximum number of frames any word has. Padding is
    # filled with duplicates of the last frame.
    for i, f in enumerate(x):
        last = np.array(f[-1].reshape(1,512))
        for _ in xrange(f.shape[0], word_max_frames):
            x[i] = np.concatenate((x[i], last), axis=0)

    # Split into train and test data sets. This also converts to numpy arrays.
    train, test = split_train_test(x, y, train_fraction)

    return train, test, vocab


def progress_msg(stdscr, video_count, word_count, video_name, num_videos):
    stdscr.addstr(1, 0, 'Processed {}/{} videos; {} words.'.format(video_count,
                                                                   num_videos,
                                                                   word_count))
    stdscr.addstr(2, 0, 'Processing {}...'.format(video_name))
    stdscr.refresh()



def main(speaker):
    ''' Progressively convert video data into facial feature data and save it. '''
    stdscr = curses_init()

    # Create facial recognition model.
    input_tensor = Input(shape=(224, 224, 3))
    vgg_model = VGGFace(input_tensor=input_tensor, include_top=False, pooling='avg')

    # Get list of all videos to process.
    video_glob = os.path.join(VIDEO_DIRECTORY_PATH, speaker, '*.mpg')
    video_paths = glob.glob(video_glob)
    num_videos = len(video_paths)
    word_count = 0

    stdscr.addstr(0, 0, 'Extracting facial features for speaker {}. Press q to exit.'.format(speaker))
    stdscr.refresh()

    try:
        for video_count, video_path in enumerate(video_paths):
            video_name = os.path.basename(video_path)
            progress_msg(stdscr, video_count, word_count, video_name, num_videos)

            word_frames, output = process_video(speaker, video_name)

            name_no_ext = video_name.split('.')[0]

            # Process each word's set of frames.
            for i, word_frame in enumerate(word_frames):
                # Check if the user has hit q to exit.
                c = stdscr.getch()
                if c == ord('q'):
                    raise SystemExit

                word_count += 1
                progress_msg(stdscr, video_count, word_count, video_name, num_videos)

                # Format of the file name is [video_name]_[word_index]_[word].
                feature_file_name = '{}_{}_{}'.format(name_no_ext, i, output[i])
                feature_file_path = os.path.join(FEATURE_DIRECTORY_PATH, speaker, feature_file_name)

                # If the file already exists, we don't want to waste time processing it again.
                if os.path.isfile(feature_file_path + '.npy'):
                    continue

                # Classify the frames and save the features to a file.
                features = vgg_model.predict(word_frame)
                np.save(feature_file_path, features)
    except SystemExit, KeyboardInterrupt:
        # Expected exceptions are either the generic one raised when a user
        # presses 'q' to exit, or a KeyboardInterrupt if the user gets
        # impatient and presses Ctrl-C.
        pass
    finally:
        curses_clean_up()


# When run as a script, the program processes video to extract facial features
# and save them for later.
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: Must specify speaker.')
    else:
        main(sys.argv[1])

