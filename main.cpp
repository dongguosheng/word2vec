#include <iostream>
#include <vector>
#include "word2vec.h"
#include <sys/time.h>
#include <cstdio>

int main() {
    using namespace std;
    using namespace w2v;

    const char *filename = "data.txt";
    int n_dim = 64;
    int n_window = 2;
    int min_cnt = 5;
    float sample = 10e8;
    int sg = 1;
    int n_neg = 10;
    float lr = 0.025;
    int n_iter = 1;
    Word2Vec w2v(filename, n_dim, n_window, min_cnt, sample, sg, n_neg, lr, n_iter);
    w2v.BuildVocab();
    cout << "Begin to Train ..." << endl;
    struct timeval st; gettimeofday( &st, NULL );
    w2v.Train();
    struct timeval et; gettimeofday( &et, NULL );
    printf("timeval: %f Seconds.\n", ((et.tv_sec - st.tv_sec) * 1000 + (et.tv_usec - st.tv_usec)/1000) / 1000.0);
    const char *output = "test";
    w2v.Save(output);
    
    return 0;
}
