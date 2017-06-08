# -*- coding: utf-8 -*-

from __future__ import print_function
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, Activation, Input, LSTM, Bidirectional
from keras.models import Sequential
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate, average, maximum
from keras.constraints import max_norm
from keras.regularizers import l2
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
from heapq import nlargest

import os
import numpy as np
import codecs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train = False
submit = True

np.random.seed(1337)
MAX_SEQUENCE_LENGTH_WORD = 25
MAX_SEQUENCE_LENGTH_CHAR = 40
MAX_NB_WORDS = 20000
MAX_NB_CHARS = 20000
EMBEDDING_DIM = 300
DATA_DIR_WORD = 'data/word'
DATA_DIR_CHAR = 'data/char'

train_texts_word = []
train_labels = []
valid_texts_word = []
valid_labels = []

train_texts_char = []
valid_texts_char = []

test_texts_word = []
test_texts_char = []

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

index2label = dict((v, k) for k, v in labels_index.iteritems())


def mixed_model():
    word_input = Input(shape=(MAX_SEQUENCE_LENGTH_WORD, ), dtype='int32')
    word_embedding_layer = Embedding(nb_words_word + 1,
                                     EMBEDDING_DIM,
                                     weights=[word_embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH_WORD,
                                     trainable=True)

    word_conv_layer = Conv1D(filters=1000, kernel_size=3, padding='valid', activation='relu', strides=1)
    word_pooling_layer = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_WORD - 3 + 1)
    word_lstm_layer = LSTM(100)

    word_level = word_embedding_layer(word_input)
    word_level = word_conv_layer(word_level)
    word_level = word_pooling_layer(word_level)
    word_level = word_lstm_layer(word_level)

    char_input = Input(shape=(MAX_SEQUENCE_LENGTH_CHAR, ), dtype='int32')
    char_embedding_layer = Embedding(nb_words_char + 1,
                                     EMBEDDING_DIM,
                                     weights=[char_embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH_CHAR,
                                     trainable=True)

    char_conv_layer = Conv1D(filters=1000, kernel_size=3, padding='valid', activation='relu', strides=1)
    char_pooling_layer = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH_CHAR - 3 + 1)
    char_lstm_layer = LSTM(100)

    char_level = char_embedding_layer(char_input)
    char_level = char_conv_layer(char_level)
    char_level = char_pooling_layer(char_level)
    char_level = char_lstm_layer(char_level)

    mixed_layer = concatenate(inputs=[word_level, char_level])
    mixed_output = Dense(len(labels_index), activation='softmax')(mixed_layer)

    model = Model(inputs=[word_input, char_input], outputs=mixed_output)
    return model

if __name__ == '__main__':

    print('Indexing vectors.')

    pre_trained_word_embeddings = Word2Vec.load('nlpcc_task2_300_dim.bin')
    pre_trained_char_embeddings = Word2Vec.load('nlpcc_task2_char_300_dim.bin')

    word_weights = pre_trained_word_embeddings.wv.syn0
    char_weights = pre_trained_char_embeddings.wv.syn0
    word_embeddings_index = dict([(k, v.index) for k, v in pre_trained_word_embeddings.wv.vocab.items()])
    char_embeddings_index = dict([(k, v.index) for k, v in pre_trained_char_embeddings.wv.vocab.items()])

    print('Found %s word vectors.' % len(word_embeddings_index))
    print('Found %s char vectors.' % len(char_embeddings_index))

    print('Processing word text dataset')

    with codecs.open(os.path.join(DATA_DIR_WORD, 'train.txt'), 'rb') as f:
        for line in f.readlines():
            train_texts_word.append(line.strip().split('\t')[1])
            train_labels.append(labels_index[line.strip().split('\t')[0]])

    with codecs.open(os.path.join(DATA_DIR_WORD, 'dev.txt'), 'rb') as f:
        for line in f.readlines():
            valid_texts_word.append(line.strip().split('\t')[1])
            valid_labels.append(labels_index[line.strip().split('\t')[0]])

    with codecs.open(os.path.join(DATA_DIR_WORD, 'test.txt'), 'rb') as f:
        for line in f.readlines():
            test_texts_word.append(line.strip())

    print('Processing char text dataset')

    with codecs.open(os.path.join(DATA_DIR_CHAR, 'train.txt'), 'rb') as f:
        for line in f.readlines():
            train_texts_char.append(line.strip().split('\t')[1])

    with codecs.open(os.path.join(DATA_DIR_CHAR, 'dev.txt'), 'rb') as f:
        for line in f.readlines():
            valid_texts_char.append(line.strip().split('\t')[1])

    with codecs.open(os.path.join(DATA_DIR_CHAR, 'test.txt'), 'rb') as f:
        for line in f.readlines():
            test_texts_char.append(line.strip())

    print('Found %s train texts. (word)' % len(train_texts_word))
    print('Found %s valid texts. (word)' % len(valid_texts_word))
    print('Found %s test texts. (word)' % len(test_texts_word))

    print('Found %s train texts. (char)' % len(train_texts_char))
    print('Found %s valid texts. (char)' % len(valid_texts_char))
    print('Found %s test texts. (char)' % len(test_texts_char))

    tokenizer_word = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer_word.fit_on_texts(train_texts_word)
    train_sequences_word = tokenizer_word.texts_to_sequences(train_texts_word)
    valid_sequences_word = tokenizer_word.texts_to_sequences(valid_texts_word)
    test_sequences_word = tokenizer_word.texts_to_sequences(test_texts_word)

    tokenizer_char = Tokenizer(num_words=MAX_NB_CHARS)
    tokenizer_char.fit_on_texts(train_texts_char)
    train_sequences_char = tokenizer_char.texts_to_sequences(train_texts_char)
    valid_sequences_char = tokenizer_char.texts_to_sequences(valid_texts_char)
    test_sequences_char = tokenizer_char.texts_to_sequences(test_texts_char)

    word_index_word = tokenizer_word.word_index
    word_index_char = tokenizer_char.word_index
    print('Found %s unique tokens. (word)' % len(word_index_word))
    print('Found %s unique tokens. (char)' % len(word_index_char))

    x_train_word = pad_sequences(train_sequences_word, maxlen=MAX_SEQUENCE_LENGTH_WORD)
    x_valid_word = pad_sequences(valid_sequences_word, maxlen=MAX_SEQUENCE_LENGTH_WORD)
    x_test_word = pad_sequences(test_sequences_word, maxlen=MAX_SEQUENCE_LENGTH_WORD)

    x_train_char = pad_sequences(train_sequences_char, maxlen=MAX_SEQUENCE_LENGTH_CHAR)
    x_valid_char = pad_sequences(valid_sequences_char, maxlen=MAX_SEQUENCE_LENGTH_CHAR)
    x_test_char = pad_sequences(valid_sequences_char, maxlen=MAX_SEQUENCE_LENGTH_CHAR)

    y_train = to_categorical(np.asarray(train_labels))
    y_valid = to_categorical(np.asarray(valid_labels))
    print('Shape of train data tensor: (word)', x_train_word.shape)
    print('Shape of train data tensor: (char)', x_train_char.shape)
    print('Shape of train label tensor:', y_train.shape)
    print('Shape of valid data tensor: (word)', x_valid_word.shape)
    print('Shape of valid data tensor: (char)', x_valid_char.shape)
    print('Shape of valid label tensor:', y_valid.shape)
    print('Shape of test data tensor: (word)', x_test_word.shape)
    print('Shape of test data tensor: (char)', x_test_char.shape)

    print('Preparing embedding matrix.')

    nb_words_word = min(MAX_NB_WORDS, len(word_index_word))
    nb_words_char = min(MAX_NB_WORDS, len(word_index_char))
    word_embedding_matrix = np.zeros((nb_words_word + 1, EMBEDDING_DIM))
    char_embedding_matrix = np.zeros((nb_words_char + 1, EMBEDDING_DIM))

    for word, i in word_index_word.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = word_embeddings_index.get(word.decode('utf-8'))
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            word_embedding_matrix[i] = word_weights[word_embeddings_index[word.decode('utf-8')], :]

    for char, i in word_index_char.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = char_embeddings_index.get(char.decode('utf-8'))
        if embedding_vector is not None:
            # chars not found in embedding index will be all-zeros.
            char_embedding_matrix[i] = char_weights[char_embeddings_index[char.decode('utf-8')], :]

    print('Training model.')

    check_pointer = ModelCheckpoint(filepath='models/best_weights.hdf5',
                                    verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=2)

    final_model = mixed_model()
    final_model.compile(loss='categorical_crossentropy',
                        optimizer='adagrad',
                        metrics=['accuracy'])
    if train:
        final_model_hist = final_model.fit(x=[x_train_word, x_train_char], y=y_train,
                                           validation_data=([x_valid_word, x_valid_char], y_valid),
                                           epochs=100,
                                           batch_size=128,
                                           callbacks=[early_stopping, check_pointer])

        print('Results Summary')
        print('*' * 20)
        print('Best Accuracy:', np.max(final_model_hist.history['val_acc']))
        print('*' * 20)

    if submit:
        final_model.load_weights('models/best_weights.hdf5')
        # y_test = final_model.predict(x=[x_test_word, x_test_char])
        y_test = final_model.predict(x=[x_valid_word, x_valid_char])

        result_lines = []
        with codecs.open('dev-top3-res.txt', mode='wb') as submit_file:
            for idx, item in enumerate(y_test):
                top_3 = nlargest(3, range(len(labels_index)), key=lambda j: item[j])
                # line = index2label[top_3[0]] + '\t' + str(item[top_3[0]]) + '\t' + index2label[top_3[1]] + '\t'\
                #     + index2label[top_3[2]] + '\t' + test_texts_word[idx] + '\n'
                line = index2label[top_3[0]] + '\t' + index2label[top_3[1]] + '\t' + index2label[top_3[2]] + '\t' +\
                       valid_texts_word[idx] + '\n'
                result_lines.append(line)
            submit_file.writelines(result_lines)
