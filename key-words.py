# -*- coding:utf-8 -*-
import jieba
import jieba.analyse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
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
    # file_path = 'data/text-label/'
    # fr_dict = dict()
    # for label in labels_index.keys():
    #     fr = open('data/text-label/' + label + '-train.txt', 'r')
    #     fr_dict.update({label: fr})
    # topK = 50
    # key_words_writer = open('key-words.txt', 'w')
    # for label in labels_index.keys():
    #     data_reader = fr_dict.get(label).read()
    #     tags = jieba.analyse.extract_tags(data_reader, topK=topK)
    #     key_words_writer.write(str(label)+'\t'+str(' '.join(tags))+'\n')
    # print "done!"
    # fr = open('key-words.txt', 'r')
    # fw = open('new-key-words.txt', 'w')
    # words_list = []
    # for line in fr.readlines():
    #     line_list = line.split('\t')
    #     label = line_list[0]
    #     str_line = str(line_list[1])
    #     str_list = str_line.split(' ')
    #     for word in str_list:
    #         words_list.append(word)
    # fr.close()
    # fr = open('key-words.txt', 'r')
    # for line in fr.readlines():
    #     line_list = line.split('\t')
    #     label = line_list[0]
    #     str_line = str(line_list[1])
    #     str_list = str_line.split(' ')
    #     label_word = []
    #     for word in str_list:
    #         count = 0
    #         for s in words_list:
    #             if word == s:
    #                 count += 1
    #         if count == 1:
    #             label_word.append(word)
    #     fw.write(label+'\t'+'\t'.join(label_word))
    fr = open('data/word/train+dev.txt', 'r')
    fw = open('data/word/new-train+dev.txt', 'w')
    while True:
        str_line = fr.readline()
        if not str_line:
            break
        str_list = str_line.split('\t')
        fw.write(''.join(str_list[1:]))
