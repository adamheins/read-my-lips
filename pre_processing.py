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

from vocab import Vocabulary


VIDEO_DIRECTORY_PATH = 'data/videos'
ALIGN_DIRECTORY_PATH = 'data/align'
FEATURE_DIRECTORY_PATH = 'data/features'

DEFAULT_TRAINING_FRACTION = 0.7

# Number of frames to use for each word.
NUM_FRAMES = 6

# Number of facial features to use. Maximum is 512.
NUM_FACIAL_FEATURES = 512


# Parse alignment file into list of (start, end, word) tuples.
def parse_alignment(fp):
    ''' Parse alignment data for an align file. '''
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
    ''' Split a video into sets of frames corresponding to each spoken word. '''
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


# class Vocabulary(object):
#     ''' A two-way mapping between words in our vocabulary and indexes. This
#         means that one can look up a word by index or an index by word. '''
#     def __init__(self, words):
#         self.word_to_index_map = {}
#         self.index_to_word_map = {}
#
#         word_set = set(words)
#
#         for index, word in enumerate(word_set):
#             self.word_to_index_map[word] = index
#             self.index_to_word_map[index] = word
#
#         self.length = len(word_set)
#
#
#     def __len__(self):
#         return self.length
#
#
#     def __getitem__(self, key):
#         try:
#             return self.index_to_word_map[key]
#         except KeyError:
#             return self.word_to_index_map[key]


def build_vocab(y):
    ''' Build vocabulary and use it to format labels. '''
    vocab = Vocabulary(y)

    # Map output to numbers.
    output_vector = []
    for word in y:
        zeros = np.zeros(len(vocab), dtype=np.float32)
        zeros[vocab[word]] = 1.0
        output_vector.append(zeros)

    return vocab, output_vector


def split_train_test(x, y, k):
    ''' Split testing and training data k ways to enable k-fold
        cross-validation. '''
    interval = int(x.shape[0] / k)

    data = []
    for i in xrange(1, k):
        # Create mask for testing section of data.
        test_mask = np.zeros(x.shape[0], dtype=bool)
        test_mask[(i-1)*interval:i*interval] = True

        x_test = x[test_mask, ...]
        y_test = y[test_mask, ...]
        test = (x_test, y_test)

        x_train = x[~test_mask, ...]
        y_train = y[~test_mask, ...]
        train = (x_train, y_train)

        data.append((train, test))

    # Final interval to the end of the data. Not included in the loop in case
    # the data set could not be divided evenly into k sections.
    x_test = x[-interval:, ...]
    y_test = y[-interval:, ...]
    test = (x_test, y_test)

    x_train = x[:-interval, ...]
    y_train = y[:-interval, ...]
    train = (x_train, y_train)

    data.append((train, test))

    return data


def condense_frames(frames, desired_length):
    ''' Condense a set of frames down to a desired length by averaging
        neighbouring frames. '''

    # Already at or below desired length, nothing to be done.
    if len(frames) <= desired_length:
        return frames

    # We need to get one frame from every cond_ratio frames.
    cond_ratio = len(frames) * 1.0 / desired_length

    condensed_frames = np.zeros((desired_length, frames.shape[1]), dtype=frames.dtype)

    frame_count = 0.0

    for i in xrange(desired_length):
        idx = int(frame_count)
        end_idx = idx + 1
        weights = [idx + 1 - frame_count]

        # Frames that are contributing fully to this average have a weight of 1.
        while cond_ratio - sum(weights) >= 1.0:
            weights.append(1.0)
            end_idx += 1

        # Account for any remainder.
        if cond_ratio > sum(weights) + 0.0000001:
            weights.append(cond_ratio - sum(weights))
            end_idx += 1

        # Normalize the weights.
        norm = np.linalg.norm(weights)
        weights = [w / norm for w in weights]

        condensed_frames[i,:] = np.average(frames[idx:end_idx,:], axis=0, weights=weights)

        frame_count += cond_ratio

    return condensed_frames


def load_data(num_words=0, k=4, speakers=[]):
    ''' Load facial feature data from disk. '''
    # If num_words is greater than 0, only that many word files with be used as
    # input. Otherwise, all available will be used.

    x = [] # Input
    y = [] # Labels

    # Select data files to load. Loads data from speakers specified, or takes
    # all data is no speakers are specified.
    if len(speakers) == 0:
        data_glob = os.path.join(FEATURE_DIRECTORY_PATH, '*', '*.npy')
        data_files = glob.glob(data_glob)
    else:
        data_files = []
        for speaker in speakers:
            data_glob = os.path.join(FEATURE_DIRECTORY_PATH, speaker, '*.npy')
            data_files.extend(glob.glob(data_glob))

    # Limit to a certain number of words, if specified.
    if num_words != 0 and len(data_files) > num_words:
        data_files = data_files[0:num_words]

    # Load the data.
    empty_file_count = 0
    for data_file in data_files:
        name_no_ext = os.path.basename(data_file).split('.')[0]
        word = name_no_ext.split('_')[2]

        data = np.load(data_file)

        # Some videos are corrupted, leading to empty data files. Skip these.
        if data.shape[0] == 0:
            empty_file_count += 1
            continue

        x.append(data[:,:NUM_FACIAL_FEATURES])
        y.append(word)
    print('Skipped {} empty data files.'.format(empty_file_count))

    # Build mapping of vocabulary to integers, and remap output to it.
    vocab, y = build_vocab(y)

    # Calculate the maximum number of frames any word has.
    word_max_frames = 0
    for word in x:
        if word.shape[0] > word_max_frames:
            word_max_frames = word.shape[0]

    # Create a mask to remove frames beyond a certain number.
    # mask = np.ones(word_max_frames, dtype=bool)
    mask = np.ones(NUM_FRAMES+1, dtype=bool)
    mask[NUM_FRAMES:] = False

    for i, f in enumerate(x):
        last = np.array(f[-1].reshape(1, NUM_FACIAL_FEATURES))

        # Add padding with duplicates of last frame.
        for _ in xrange(f.shape[0], NUM_FRAMES + 1):
            x[i] = np.concatenate((x[i], last), axis=0)

        x[i] = condense_frames(x[i], NUM_FRAMES + 1)

        # Take deltas.
        for j in xrange(1, NUM_FRAMES + 1):
            x[i][j-1,:] = x[i][j,:] - x[i][j-1,:]

        # Apply mask to remove extra frames.
        x[i] = x[i][mask, ...]

        # Normalize the entire set of frames for this word.
        norm = np.linalg.norm(x[i])
        if norm == 0:
            # If the norm is zero, meaning the vector is zero, we just use an
            # evenly distributed unit array.
            ones = np.ones(x[i].shape)
            x[i] = ones / np.linalg.norm(ones)
        else:
            x[i] = x[i] / norm

    x = np.asarray(x)
    y = np.asarray(y)

    # Split into train and test data sets. This also converts to numpy arrays.
    data = split_train_test(x, y, k)

    return data, vocab


def progress_msg(stdscr, video_count, word_count, video_name, num_videos):
    ''' Display preprocessing progress message. '''
    stdscr.addstr(1, 0, 'Processed {}/{} videos; {} words.'.format(video_count,
                                                                   num_videos,
                                                                   word_count))
    stdscr.addstr(2, 0, 'Processing {}...'.format(video_name))
    stdscr.refresh()


def curses_init():
    ''' Initialize curses interface. '''
    stdscr = curses.initscr()
    stdscr.nodelay(True)
    curses.noecho()
    curses.cbreak()
    return stdscr


def curses_clean_up():
    ''' Clean up curses interface. '''
    curses.echo()
    curses.nocbreak()
    curses.endwin()


def main(speaker):
    ''' Progressively convert video data into facial feature data and save it. '''
    stdscr = curses_init()

    # Create facial recognition model.
    input_tensor = Input(shape=(224, 224, 3))
    vgg_model = VGGFace(input_tensor=input_tensor, include_top=False, pooling='avg')

    # Create the speaker's feature directory if it doesn't already exist.
    speaker_feature_dir = os.path.join(FEATURE_DIRECTORY_PATH, speaker)
    if not os.path.isdir(speaker_feature_dir):
        os.mkdir(speaker_feature_dir)

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
                feature_file_path = os.path.join(speaker_feature_dir, feature_file_name)

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
        print('Error: Must specify at least one speaker.')
        print('python pre_processing.py speaker1 [speaker2] ...')
    else:
        for speaker in sys.argv[1:]:
            main(speaker)

