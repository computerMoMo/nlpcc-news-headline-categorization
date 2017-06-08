# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import sys

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

index_labels = {0: 'history',
                1: 'military',
                2: 'baby',
                3: 'world',
                4: 'tech',
                5: 'game',
                6: 'society',
                7: 'sports',
                8: 'travel',
                9: 'car',
                10: 'food',
                11: 'entertainment',
                12: 'finance',
                13: 'fashion',
                14: 'discovery',
                15: 'story',
                16: 'regimen',
                17: 'essay'}

if __name__ == '__main__':
    fw_dict = dict()
    for label in labels_index.keys():
        fw = open('data/text-label/' + label + '-train.txt', 'w')
        fw_dict.update({label: fw})
    data_reader = open('data/word/train.txt', 'r')
    print('analyse')

    while True:
        str_line = data_reader.readline()
        if not str_line:
            break
        str_list = str_line.split('\t')
        label = str_list[0]

        fw_dict.get(str(label)).write(str(str_list[1]))
