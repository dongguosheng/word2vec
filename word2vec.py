# -*- coding: gbk -*-

import numpy as np
import sys
from operator import itemgetter

class Word2Vec(object):
    def __init__(self, model_name):
        self.word2idx = {}
        self.vocab_list = []
        self.n_dim = 0
        with open('{}_vocab.txt'.format(model_name)) as fin:
            self.n_dim = int(fin.next())
            for line in fin:
                if len(line.rstrip().split()) != 3:
                    continue
                word, index, cnt = line.rstrip().split()
                self.word2idx[word] = int(index)
                self.vocab_list.append( (word, int(index), int(cnt)) )
        data = np.fromfile('{}_syn0.bin'.format(model_name), dtype='float32')
        self.n_row = int(data.shape[0] / self.n_dim)
        self.syn0 = np.asarray(data).reshape(self.n_row, self.n_dim)
    def similarity(self, word_1, word_2):
        if word_1 not in self.word2idx:
            print '%s not in vocab.' % word_1
            return None
        if word_2 not in self.word2idx:
            print '%s not in vocab.' % word_2
            return None
        vec1 = self.syn0[self.word2idx[word_1]]
        print 'vec1 norm: %f' % np.linalg.norm(vec1)
        vec2 = self.syn0[self.word2idx[word_2]]
        print 'vec2 norm: %f' % np.linalg.norm(vec2)
        sim = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
        return sim
    def most_similar(self, word, topk=10):
        rs_list = []
        if word not in self.word2idx:
            print '%s not in vocab.' % word
            return None
        vec = self.syn0[self.word2idx[word]]
        for idx in xrange(self.n_row):
            sim = np.dot(vec, self.syn0[idx]) / np.linalg.norm(vec) / np.linalg.norm(self.syn0[idx])
            rs_list.append( (idx, sim) )
        rs_list = sorted(rs_list, key=itemgetter(1), reverse=True)
        return rs_list
    def print_most_similar(self, word, topk=10):
        rs_list = self.most_similar(word, topk=topk)
        for idx, sim in rs_list[: topk]:
            print self.vocab_list[idx][0], sim

def main():
    if len(sys.argv) == 3:
        m = Word2Vec('test')
        print m.similarity(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        m = Word2Vec('test')
        m.print_most_similar(sys.argv[1])
    else:
        print 'not supported.'

if __name__ == '__main__':
    main()
