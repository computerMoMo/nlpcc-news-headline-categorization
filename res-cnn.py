# -*- coding: utf-8 -*-
from __future__ import print_function
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, Activation, Input, LSTM, Bidirectional
from keras.models import Sequential
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate
from keras.constraints import max_norm
from keras.regularizers import l2

import os
import sys
import numpy as np
import codecs

if __name__ == '__main__':
    x_train = np.load('data/new-x-train-res.npy')
    y_train = np.load('data/new-y-train-res.npy')
    x_test = np.load('data/new-x-test-res.npy')
    y_test = np.load('data/new-y-test-res.npy')

    y_train = to_categorical(np.asarray(y_train))
    y_test = to_categorical(np.asarray(y_test))

    model = Sequential()
    model.add(Embedding(19, 50, input_length=5, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=2, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=model.output_shape[1]))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(18, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, validation_data=[x_test, y_test], epochs=5, batch_size=128)
    scores = model.evaluate(x=x_test, y=y_test, batch_size=128)
    print(scores)
