# -*- coding: gbk -*-

import numpy as np
import sys
from operator import itemgetter
import os

class Query2Vec(object):
    def __init__(self, model_name):
        self.word2idx = {}
        self.term2idx = {}
        self.ngram2idx = {}
        self.vocab_list = []
        self.term_list = []
        self.n_dim = 0
        with open('{}_vocab.txt'.format(model_name)) as fin:
            self.n_dim = int(fin.next())
            for line in fin:
                if len(line.rstrip().split()) != 3:
                    continue
                word, index, cnt = line.rstrip().split()
                self.word2idx[word] = int(index)
                self.vocab_list.append( (word, int(index), int(cnt)) )
        with open('{}.term.idxlist'.format(model_name)) as fin:
            for line in fin:
                tmp_list = line.rstrip().split()
                if len(tmp_list) != 2:
                    continue
                term, term_idx = tmp_list
                self.term2idx[term] = int(term_idx)
                self.term_list.append( (term, int(term_idx)) )
        data = np.fromfile('{}_syn0.bin'.format(model_name), dtype='float32')
        self.n_row = int(data.shape[0] / self.n_dim)
        print 'Total %d nodes.' % self.n_row
        self.syn0 = np.asarray(data).reshape(self.n_row, self.n_dim)
        data = np.fromfile('{}_syn1neg.bin'.format(model_name), dtype='float32')
        self.syn1neg = np.asarray(data).reshape(self.n_row, self.n_dim)
        data = np.fromfile('{}_syn0_term.bin'.format(model_name), dtype='float32')
        n_term = int(data.shape[0] / self.n_dim)
        self.syn0_term = np.asarray(data).reshape(n_term, self.n_dim)
        if (os.path.exists('{}_ngram.txt'.format(model_name))):
            with open('{}_ngram.txt'.format(model_name)) as fin:
                for line in fin:
                    tmp_list = line.rstrip().split('\t')
                    if len(tmp_list) != 2:
                        continue
                    ngram, ngram_idx = tmp_list
                    self.ngram2idx[ngram] = int(ngram_idx)
            data = np.fromfile('{}_syn0_ngram.bin'.format(model_name), dtype='float32')
            n_ngram = int(data.shape[0] / self.n_dim)
            self.syn0_ngram = np.asarray(data).reshape(n_ngram, self.n_dim)
    def seg(self, s):
        return s.split()
    def get_ngrams(self, l, n):
        ngrams = []
        for i in range(2, n+1):
            for j in range(len(l) - i + 1):
                ngrams.append( ' '.join(l[j : j+i]) )
        ngrams.append(l[-1] + '#')
        if len(l) > 1:
            ngrams.append(l[-2]+l[-1]+'#')
        return ngrams
    def get_embedding(self, word):
        if word not in self.word2idx:
            print '%s not in vocab.' % word
            term_list = self.seg(word)
            print 'with Query: ' + '/'.join([term for term in term_list if term in self.term2idx])
            ngrams = self.get_ngrams(term_list, 2)
            vec_term = self.syn0_term[ [self.term2idx[term] for term in term_list if term in self.term2idx] ]
            print ','.join(ngrams)
            print 'with Ngram: ' + '/'.join([ngram for ngram in ngrams if ngram in self.ngram2idx])
            ngrams = [self.ngram2idx[ngram] for ngram in ngrams if ngram in self.ngram2idx]
            vec_gram = ''
            if (len(ngrams) > 0):
                vec_ngram = self.syn0_ngram[ [self.ngram2idx[ngram] for ngram in ngrams if ngram in self.ngram2idx] ]
                vec = np.append(vec_term, vec_ngram, axis=0)
            else:
                vec = vec_term
            if vec.shape[0] > 0:
                return np.average(vec, axis=0)
            else:
                return None
        else:
            return self.syn0[self.word2idx[word]]
    def similarity(self, word_1, word_2):
        vec1 = self.get_embedding(word_1)
        vec2 = self.get_embedding(word_2)
        if vec1 is None or vec2 is None:
            return -100
        print 'vec1 norm: %f' % np.linalg.norm(vec1)
        print 'vec2 norm: %f' % np.linalg.norm(vec2)
        sim = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
        return sim
    def most_similar(self, word, topk=10, is_syn1neg=False):
        rs_list = []
        vec = self.get_embedding(word)
        if vec is None:
            return []
        for idx in xrange(self.n_row):
            sim = np.dot(vec, self.syn0[idx]) / np.linalg.norm(vec) / np.linalg.norm(self.syn0[idx])
            rs_list.append( (idx, sim) )
        rs_list = sorted(rs_list, key=itemgetter(1), reverse=True)[: topk]
        return rs_list
    def print_most_similar(self, word, topk=10):
        rs_list = self.most_similar(word, topk=topk)
        for idx, sim in rs_list:
            print self.vocab_list[idx][0], sim

def main():
    if len(sys.argv) == 3:
        m = Query2Vec('test')
        print m.similarity(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        m = Query2Vec('test')
        m.print_most_similar(sys.argv[1])
    else:
        print 'not supported.'

if __name__ == '__main__':
    main()
