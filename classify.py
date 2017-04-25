from __future__ import print_function

import numpy as np
np.random.seed(1337)

from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional

from pre_processing import load_data, split_train_test

# Data parameters.
SPEAKERS = range(1, 16) + [17, 18] + range(27, 35)
K = len(SPEAKERS)
print('Speakers: ', K)
SHUFFLE = False
USE_DELTA_FRAMES = False

# Model parameters.
EPOCHS = 30 # 50 for delta frames, 80 without
BATCH_SIZE = 32
ACTIVATION = 'tanh'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'

VERBOSE = True


# Load the data.
x, y, kmasks, vocab = load_data(k=K, speakers=SPEAKERS, shuffle=SHUFFLE,
                                use_delta_frames=USE_DELTA_FRAMES)

accuracies = []
epochs = []

for fold in xrange(1,3):
    print('Fold: ', fold)

    # Split data based on current fold.
    train, test = split_train_test(x, y, kmasks[fold])
    x_train, y_train = train
    x_test, y_test = test

    frames_per_word = x_train.shape[1]
    features_per_frame = x_train.shape[2]

    if VERBOSE:
        print('Number of words for training: ', x_train.shape[0])
        print('Number of words for testing: ', x_test.shape[0])
        print('Frames per word: ', frames_per_word)
        print('Features per frame: ', features_per_frame)

    # Build the LSTM model.
    model = Sequential()
    model.add(LSTM(128, input_shape=x_train[0].shape, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(len(vocab), activation=ACTIVATION))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])

    # Train the model.
    accs = []
    for _ in xrange(EPOCHS):
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1,
                  verbose=VERBOSE, validation_data=(x_test, y_test))
        _, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE,
                                verbose=VERBOSE)
        accs.append(acc)

    # Find epoch with best accuracy.
    best_acc = 0
    best_epoch = 0
    for epoch, acc in enumerate(accs):
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

    print('Best accuracy: {} at epoch {}.'.format(best_acc, best_epoch))
    accuracies.append(best_acc)
    epochs.append(best_epoch)

print('Accuracies: ', accuracies)
print('Epochs: ', epochs)

# Statistics.
# print('Mean Acc:  ', np.mean(accuracies))
# print('Stdev Acc: ', np.std(accuracies))
# print('Mean epochs: ', np.mean(epochs))


# correct_occurrences = {}
# for test_input, test_label in zip(x_test, y_test):
#     pred = model.predict(test_input.reshape(1, frames_per_word, features_per_frame))
#     pred_idx = np.argsort(pred.reshape(len(vocab)))[::-1][0]
#     corr_idx = np.argsort(test_label)[::-1][0]
#
#     pred_word = vocab[pred_idx]
#     corr_word = vocab[corr_idx]
#     match = pred_word == corr_word
#
#     if pred_word == corr_word:
#     if corr_word in results.keys()
#
#     results[corr_word] 
