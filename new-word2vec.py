#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line script for generating embeddings
Useful if you want to generate larger embeddings for some models
"""

from __future__ import print_function
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import os
import sys
import random
import argparse
import logging
import codecs

random.seed(1337)


# parse arguments
parser = argparse.ArgumentParser(description='Generate embeddings for the NLPCC-Task2 dataset')
parser.add_argument('--iter', metavar='N', type=int, default=10, help='number of times to run')
parser.add_argument('--size', metavar='D', type=int, default=300, help='dimensions in embedding')
parser.add_argument('--str', type=str)
args = parser.parse_args()
data_path = 'data/'+args.str
# configure logging
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

# prepare corpus
sentences = LineSentence(os.path.join(data_path, 'train+dev.txt'))

# run model
model = Word2Vec(sentences, size=args.size, min_count=2, window=5, sg=1, iter=args.iter, workers=20)
model.save('nlpcc_task2_' + args.str + '_train+dev_%d_dim.bin' % args.size)
