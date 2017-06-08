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
import numpy as np
import codecs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


S_CNN = True
M_CNN = True

U_LSTM = True
B_LSTM = True

CNN_LSTM = True

np.random.seed(1337)
MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
DATA_DIR = 'data/char'

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


def single_cnn():

    model = Sequential()

    model.add(Embedding(nb_words + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=True))

    model.add(Conv1D(filters=250,
                     kernel_size=3,
                     padding='valid',
                     activation='relu',
                     strides=1))

    model.add(MaxPooling1D(pool_size=model.output_shape[1]))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(len(labels_index), activation='softmax'))

    return model


def multi_cnn():
    nb_filter = 250
    filter_lengths = [2, 3, 5, 7]
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    embedded_sequences = embedding_layer(sequence_input)

    cnn_layers = []

    for filter_length in filter_lengths:
        x = Conv1D(filters=nb_filter,
                   kernel_size=filter_length,
                   padding='valid',
                   activation='relu',
                   kernel_constraint=max_norm(3),
                   kernel_regularizer=l2(0.0001),
                   strides=1)(embedded_sequences)
        x = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - filter_length + 1)(x)
        x = Flatten()(x)
        cnn_layers.append(x)

    x = concatenate(cnn_layers)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    y_hat = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, y_hat)

    return model


def unidirection_lstm():

    model = Sequential()

    model.add(Embedding(nb_words + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(len(labels_index), activation='softmax'))

    return model


def bidirection_lstm():

    model = Sequential()

    model.add(Embedding(nb_words + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=True))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels_index), activation='softmax'))

    return model


def cnn_lstm():
    model = Sequential()

    model.add(Embedding(nb_words + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=True))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=1000,
                     kernel_size=3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=model.output_shape[1]))
    model.add(LSTM(128))
    model.add(Dense(len(labels_index), activation='softmax'))

    return model


if __name__ == '__main__':

    print('Indexing word vectors.')

    pre_trained_embeddings = Word2Vec.load('nlpcc_task2_char_300_dim.bin')

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
    if S_CNN:

        s_cnn = single_cnn()
        s_cnn.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        s_cnn_hist = s_cnn.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                               epochs=5, batch_size=128)
        best_accuracy.append(np.max(s_cnn_hist.history['val_acc']))
        model_names.append('S_CNN')
        # save results
        # s_cnn_result_array = s_cnn.predict_classes(x_valid, batch_size=128)
        # np.save('data/s_cnn_result.npy', s_cnn_result_array)
        # print('s_cnn result shape:', s_cnn_result_array.shape)

    if M_CNN:
        m_cnn = multi_cnn()
        m_cnn.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        m_cnn_hist = m_cnn.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                               epochs=5, batch_size=128)
        best_accuracy.append(np.max(m_cnn_hist.history['val_acc']))
        model_names.append('M_CNN')
        # save results
        # m_cnn_result_array = m_cnn.predict(x_valid, batch_size=128)
        # m_cnn_result_classes = [np.argmax(class_list) for class_list in m_cnn_result_array]
        # m_cnn_result_classes_array = np.asarray(m_cnn_result_classes, dtype=np.int8)
        # np.save('data/m_cnn_result.npy', m_cnn_result_classes_array)
        # print('m_cnn result shape:', m_cnn_result_classes_array.shape)

    if U_LSTM:
        u_lstm = unidirection_lstm()
        u_lstm.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

        u_lstm_hist = u_lstm.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                                 epochs=5, batch_size=128)
        best_accuracy.append(np.max(u_lstm_hist.history['val_acc']))
        model_names.append('U_LSTM')
        # save results
        # u_lstm_result_array = u_lstm.predict_classes(x_valid, batch_size=128)
        # np.save('data/u_lstm_result.npy', u_lstm_result_array)
        # print('u_lstm result shape:', u_lstm_result_array.shape)

    if B_LSTM:
        b_lstm = bidirection_lstm()
        b_lstm.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

        b_lstm_hist = b_lstm.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                                 epochs=5, batch_size=128)
        best_accuracy.append(np.max(b_lstm_hist.history['val_acc']))
        model_names.append('B_LSTM')
        # save results
        # b_lstm_result_array = b_lstm.predict_classes(x_valid, batch_size=128)
        # np.save('data/b_lstm_result.npy', b_lstm_result_array)
        # print('b_lstm result shape:', b_lstm_result_array.shape)

    if CNN_LSTM:
        conv_lstm = cnn_lstm()
        conv_lstm.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

        conv_lstm_hist = conv_lstm.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                                       epochs=5, batch_size=128)
        best_accuracy.append(np.max(conv_lstm_hist.history['val_acc']))
        model_names.append('CNN_LSTM')
        # save results
        # conv_lstm_result_array = conv_lstm.predict_classes(x_valid, batch_size=128)
        # np.save('data/conv_lstm_result.npy', conv_lstm_result_array)
        # print('conv_lstm result shape:', conv_lstm_result_array.shape)

    # # Plot model accuracy
    # for idx, hist in enumerate(hists):
    #     plt.plot(hist.history['acc'], color='blue', label=model_names[idx]+' train')
    #     plt.plot(hist.history['val_acc'], color='red', label=model_names[idx] + ' valid')
    # plt.title('Model Accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(loc='upper left')
    # plt.savefig('accuracy.png')
    #
    # # Plot model loss
    # for idx, hist in enumerate(hists):
    #     plt.plot(hist.history['loss'], color='blue', label=model_names[idx] + ' train')
    #     plt.plot(hist.history['val_loss'], color='red', label=model_names[idx] + ' valid')
    # plt.title('Model Loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(loc='lower left')
    # plt.savefig('loss.png')

    print('Results Summary')
    assert len(model_names) == len(best_accuracy)
    for i in range(len(model_names)):
        print('*' * 20)
        print('Model Name:', model_names[i])
        print('Best Accuracy:', best_accuracy[i])
        print('*' * 20)
