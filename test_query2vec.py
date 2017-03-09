# -*- coding: gbk -*-

from query2vec import Query2Vec
import sys

# MODEL_DIR = 'model_single_thread/'
# MODEL_DIR = 'model_omp_lock/rw_lock/'
# MODEL_DIR = './model_avg_pooling/'
# MODEL_DIR = './model_avg_pooling_total/'
# MODEL_DIR = './model_avg_pooling_total/ngram/'
# MODEL_NAME = 'test.tmp'
# MODEL_NAME = 'test'
# MODEL_DIR = './'

MODEL_DIR = './model_avg_pooling_total/ngram-hash/'
MODEL_NAME = 'test.ngram.hash'
NODE_IDXLIST = '../../deepwalk_data/good_data/tmp.idxlist'
# NODE_IDXLIST = '../../deepwalk_data/good_data/test.idxlist'

def load_node_idx():
    input = NODE_IDXLIST
    idx2node = {}
    node2idx = {}
    with open(input) as fin:
        for line in fin:
            query_site, idx = line.rstrip().split('\t')
            idx2node[idx] = query_site
            node2idx[query_site] = idx
    print 'node idx file load complete.'
    return (node2idx, idx2node)

def is_site(idx):
    return True
    # return idx >= 2561687

def load_model():
    filename = MODEL_DIR + MODEL_NAME
    print 'model: ' + filename
    m = Query2Vec(filename)
    print 'query2vec load complete.'
    return m

def most_similar(query_site, node2idx, idx2node, topk=10, is_syn1neg=False):
    m = load_model()
    word = node2idx[query_site] if query_site in node2idx else query_site
    similar_idx_list = m.most_similar(word, topk=-1, is_syn1neg=is_syn1neg)
    cnt = 0
    for idx, sim in similar_idx_list:
        if is_site(int(m.vocab_list[idx][0])):
            print idx2node[m.vocab_list[idx][0]], sim
            cnt += 1
        if cnt >= topk:
            break

def similarity(query_site1, query_site2, node2idx):
    m = load_model()
    word1 = node2idx[query_site1] if query_site1 in node2idx else query_site1
    word2 = node2idx[query_site2] if query_site2 in node2idx else query_site2
    print m.similarity(word1, word2)

def main():
    if len(sys.argv) == 3:
        node2idx, idx2node = load_node_idx()
        similarity(sys.argv[1], sys.argv[2], node2idx)
    elif len(sys.argv) == 2:
        query_site = sys.argv[1]
        node2idx, idx2node = load_node_idx()
        most_similar(query_site, node2idx, idx2node, topk=25, is_syn1neg=False)
    else:
        print 'not enough args.'

if __name__ == '__main__':
    main()
