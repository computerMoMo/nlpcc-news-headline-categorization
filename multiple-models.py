# -*- coding: utf-8 -*-
from __future__ import print_function
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, Activation, Input, LSTM, Bidirectional
from keras.models import Sequential
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate, average, Conv2D, MaxPooling2D, Reshape
from keras.constraints import max_norm
from keras.regularizers import l2

import os
import sys
import numpy as np
import codecs

np.random.seed(1337)
global nb_words
global embedding_matrix

MAX_SEQUENCE_LENGTH = 25
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
CNN_FILTERS = 250
num_epochs = int(sys.argv[1])

DATA_DIR = 'data/word'

train_texts = []
train_labels = []
valid_texts = []
valid_labels = []

model_names = []
best_accuracy = []

labels_index = {'history': 0,
                'military': 1,
                'baby': 2,
                'world': 3,
                'tech': 4,
                'game': 5,
                'society': 6,
                'sports': 7,
                'travel': 8,
                'car': 9,
                'food': 10,
                'entertainment': 11,
                'finance': 12,
                'fashion': 13,
                'discovery': 14,
                'story': 15,
                'regimen': 16,
                'essay': 17}


if __name__ == '__main__':
    print('Indexing word vectors.')

    pre_trained_embeddings = Word2Vec.load('nlpcc_task2_300_dim.bin')

    weights = pre_trained_embeddings.wv.syn0
    embeddings_index = dict([(k, v.index) for k, v in pre_trained_embeddings.wv.vocab.items()])

    print('Found %s word vectors.' % len(embeddings_index))

    print('Processing text dataset')

    with codecs.open(os.path.join(DATA_DIR, 'train.txt'), 'rb') as f:
        for line in f.readlines():
            train_texts.append(line.strip().split('\t')[1])
            train_labels.append(labels_index[line.strip().split('\t')[0]])

    with codecs.open(os.path.join(DATA_DIR, 'dev.txt'), 'rb') as f:
        for line in f.readlines():
            valid_texts.append(line.strip().split('\t')[1])
            valid_labels.append(labels_index[line.strip().split('\t')[0]])

    print('Found %s train texts.' % len(train_texts))
    print('Found %s valid texts.' % len(valid_texts))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_texts)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    valid_sequences = tokenizer.texts_to_sequences(valid_texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    x_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x_valid = pad_sequences(valid_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    y_train = to_categorical(np.asarray(train_labels))
    y_valid = to_categorical(np.asarray(valid_labels))
    print('Shape of train data tensor:', x_train.shape)
    print('Shape of train label tensor:', y_train.shape)
    print('Shape of valid data tensor:', x_valid.shape)
    print('Shape of valid label tensor:', y_valid.shape)

    print('Preparing embedding matrix.')

    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word.decode('utf-8'))
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = weights[embeddings_index[word.decode('utf-8')], :]

    # build model
    model_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    layers_outputs = []

    # single CNN
    single_cnn_embedding = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    single_cnn_embedded_sequence = single_cnn_embedding(model_input)
    single_cnn_conv = Conv1D(filters=CNN_FILTERS, kernel_size=3, padding='valid',
                        activation='relu', kernel_constraint=max_norm(3), kernel_regularizer=l2(0.0001),
                        strides=1, name='single_cnn_conv')(single_cnn_embedded_sequence)
    single_cnn_maxpooling = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 3 + 1, name='single_cnn_maxpooling')(single_cnn_conv)
    single_cnn_flatten = Flatten(name='single_cnn_flatten')(single_cnn_maxpooling)
    single_cnn_dropout = Dropout(0.2, name='single_cnn_dropout')(single_cnn_flatten)
    single_cnn_dense = Dense(128, activation='relu', name='single_cnn_dense')(single_cnn_dropout)
    single_cnn_output = Dense(len(labels_index), activation='softmax', name='single_cnn_output')(single_cnn_dense)
    layers_outputs.append(single_cnn_output)

    # multiple CNN
    filter_lengths = [2, 3, 5, 7]
    multi_cnn_embedding = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH, trainable=True)
    multi_cnn_embedded_sequence = multi_cnn_embedding(model_input)
    conv_layers = []

    for filter_length in filter_lengths:
        multi_conv_layers = Conv1D(filters=CNN_FILTERS, kernel_size=filter_length, padding='valid',
                             activation='relu', kernel_constraint=max_norm(3), kernel_regularizer=l2(0.0001),
                             strides=1)(multi_cnn_embedded_sequence)
        multi_maxpooling = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - filter_length + 1)(multi_conv_layers)
        multi_flatten = Flatten()(multi_maxpooling)
        conv_layers.append(multi_flatten)

    multi_concatenate = concatenate(inputs=conv_layers)
    multi_dropout = Dropout(0.2)(multi_concatenate)
    multi_dense = Dense(128, activation='relu')(multi_dropout)
    multi_cnn_output = Dense(len(labels_index), activation='softmax', name='multi_cnn_output')(multi_dense)
    layers_outputs.append(multi_cnn_output)

    # unidirectional lstm
    uni_lstm_embedding = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                  input_length=MAX_SEQUENCE_LENGTH, trainable=True)
    uni_lstm_embedded_sequence = uni_lstm_embedding(model_input)
    uni_lstm = LSTM(units=128, activation='tanh', dropout=0.2, recurrent_dropout=0.2, name='uni_lstm')(uni_lstm_embedded_sequence)
    uni_lstm_output = Dense(len(labels_index), activation='softmax', name='uni_lstm_output')(uni_lstm)
    layers_outputs.append(uni_lstm_output)

    # bidirectional lstm
    bi_lstm_emdedding = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                  input_length=MAX_SEQUENCE_LENGTH, trainable=True)
    bi_lstm_emdedded_sequence = bi_lstm_emdedding(model_input)
    bi_lstm = Bidirectional(LSTM(128))(bi_lstm_emdedded_sequence)
    bi_lstm_dropout = Dropout(0.5)(bi_lstm)
    bi_lstm_output = Dense(len(labels_index), activation='softmax', name='bi_lstm_output')(bi_lstm_dropout)
    layers_outputs.append(bi_lstm_output)

    # cnn_lstm
    cnn_lstm_embedding = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                  input_length=MAX_SEQUENCE_LENGTH, trainable=True)
    cnn_lstm_embedded_sequence = cnn_lstm_embedding(model_input)
    cnn_lstm_conv = Conv1D(filters=CNN_FILTERS, kernel_size=3, padding='valid', activation='relu',
                           strides=1, name='c_lstm_conv')(cnn_lstm_embedded_sequence)
    cnn_lstm_maxpooling = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 3 + 1)(cnn_lstm_conv)
    cnn_lstm_lstm = LSTM(128)(cnn_lstm_maxpooling)
    cnn_lstm_output = Dense(len(labels_index), activation='softmax', name='cnn_lstm_output')(cnn_lstm_lstm)
    layers_outputs.append(cnn_lstm_output)

    # multiple model
    multiple_model_concatenate = concatenate(inputs=layers_outputs)
    multiple_model_reshape = Reshape((18*5, 1))(multiple_model_concatenate)
    multiple_model_conv = Conv1D(filters=CNN_FILTERS, kernel_size=3, padding='valid',
                             activation='relu', kernel_constraint=max_norm(3), kernel_regularizer=l2(0.0001),
                             strides=1, name='multiple_model_conv')(multiple_model_reshape)
    multiple_model_maxpooling = MaxPooling1D(pool_size=18*5-3+1, name='multiple_model_maxpooling')(
        multiple_model_conv)
    multiple_model_flatten = Flatten(name='multiple_model_flatten')(multiple_model_maxpooling)
    multiple_model_dropout = Dropout(0.2, name='multiple_model_dropout')(multiple_model_flatten)
    multiple_model_output = Dense(len(labels_index), activation='softmax', name='multiple_model_output')\
        (multiple_model_dropout)

    # train
    multiple_model = Model(inputs=model_input, outputs=multiple_model_output)
    multiple_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    multiple_model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), epochs=num_epochs, batch_size=128)
    scores = multiple_model.evaluate(x=x_valid, y=y_valid, batch_size=128)
    print('\nscores are:', scores)

    print('program finished!')
