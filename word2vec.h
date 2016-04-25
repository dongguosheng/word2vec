#ifndef WORD2VEC_H
#define WORD2VEC_H

#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdio>

namespace w2v {
    struct Vocab {
        std::string word;
        int cnt;
        size_t index;
        Vocab(std::string word, int cnt) : word(word), cnt(cnt), index(0) {}
        bool operator<(const Vocab& rhs) const {
            return rhs.cnt < cnt;
        }
    };
    struct Mat {
        size_t n_row;
        size_t n_col;
        float *dptr;
        Mat() {}
        Mat(float *dptr, size_t n_row, size_t n_col) : n_row(n_row), n_col(n_col), dptr(dptr) {}
        Mat deepcopy(const Mat &mat) {
            Mat rs;
            memcpy(rs.dptr, mat.dptr, sizeof(float) * mat.n_row * mat.n_col);
            rs.n_row = mat.n_row;
            rs.n_col = mat.n_col;
            return rs;
        }
        Mat& operator=(const Mat &mat) {
            memcpy(dptr, mat.dptr, sizeof(float) * mat.n_row * mat.n_col);
            n_row = mat.n_row;
            n_col = mat.n_col;
            return *this;
        }
        float* operator[](size_t i) {
            return dptr + i*n_col;
        }
        inline void Set(float *_dptr, size_t _n_row, size_t _n_col) {
            dptr = _dptr;
            n_row = _n_row;
            n_col = _n_col;
        }
        inline void Reset() {
            memset(dptr, 0, sizeof(float) * n_row * n_col);
        }
        inline float* GetRow(size_t i) const {
            return dptr + i;
        }
        inline void Save(const char *filename) {
            FILE *fp = fopen(filename, "wb");
            if(fp) {
                fwrite(dptr, sizeof(float), n_row * n_col, fp);
            }
            fclose(fp);
        }
    };
    class Word2Vec {
        public:
            Word2Vec() {}
            Word2Vec(const char *filename, int n_dim, int n_window, int min_cnt, float sample, int sg, int n_neg, float lr, int n_iter)
                : filename(filename), n_dim(n_dim), n_window(n_window), min_cnt(min_cnt),
                  n_sum_cnt(0), sample(sample), sg(sg), n_neg(n_neg), lr(lr), n_iter(n_iter) {
                    rng.seed(static_cast<int>(time(NULL)));
                  }
            virtual ~Word2Vec() {
                delete [] syn0.dptr;
                delete [] syn1.dptr;
            }
            inline bool BuildVocab() {
                std::unordered_map<std::string, int> raw_map;
                if(!GenRawVocab(raw_map)) {return false;}
                RefineVocab(raw_map);
                MakeCumTable(); // for negative sampling
                ResetWeight();
                std::cout << "Vocabulary Build Complete." << std::endl;
                return true;
            }
            inline void Train() {
                std::vector<std::vector<size_t> > sentences;
                LoadSentences(filename, sentences);
                // rng.seed(static_cast<int>(time(NULL)));
                std::uniform_int_distribution<int> ud(0, n_window - 1);
                for(int iter = 0; iter < n_iter; ++ iter) {
                    size_t width = sentences.size() / 1000;
                    size_t progress = 0;
                    # pragma omp parallel for
                    for(int i = 0; i < static_cast<int>(sentences.size()); ++ i) {
                        # pragma omp critical 
                        {

                            progress ++;
                            if(progress % width == 0) {
                                if (lr > 0.0001)    lr *= 0.999;
                                std::cout << "[PROGRESS";
                                std::cout << "] ##### " << (float)(progress) / sentences.size() * 100.0 << " % ### lr: " << lr << "\r" << std::flush;
                            }

                        }
                        TrainSentence(sentences[i], ud);
                    }
                }
                std::cout << std::endl;
            }
            inline void TrainSentence(const std::vector<size_t> &sentence, std::uniform_int_distribution<int> &ud) {
                if (sg > 0) {
                    for(int i = 0; i < static_cast<int>(sentence.size()); ++ i) {
                        int rand_win = ud(rng);
                        int start = std::max(0, i - n_window + rand_win);
                        int end = std::min(static_cast<int>(sentence.size()), i + n_window - rand_win + 1);
                        // std::cout << "start: " << start << ", end: " << end << std::endl;
                        for(int j = start; j < end; ++ j) {
                            if(i != j)  TrainSgPair(sentence[i], sentence[j]);
                        }
                    }
                } else {
                    
                }
            }
            inline void TrainSgPair(size_t input_index, size_t pos_index) {
                if(n_neg > 0) {
                    float *input_vec = syn0[input_index];
                    std::vector<size_t> idx_vec;
                    idx_vec.reserve(n_neg);
                    idx_vec.push_back(pos_index);
                    std::uniform_real_distribution<float> ud(0.0, cum_table[cum_table.size() - 1]);
                    NegativeSample(idx_vec, pos_index, ud);
                    // raw for loop
                    float label;
                    float *output_vec = NULL;
                    float pred_label;
                    std::vector<float> neu1e(n_dim, 0.0f);
                    float error;
                    for(int i = 0; i <= n_neg; ++ i) {
                        if (i == 0) {
                            label = 1.0f;
                        } else {
                            label = 0.0f;
                        }
                        output_vec = syn1[idx_vec[i]];
                        pred_label = 1.0 / ( 1 + exp( -DotProduct( input_vec, output_vec ) ) );
                        error = (label - pred_label) * lr;
                        for(int j = 0; j < n_dim; ++ j) {
                            // save grad for input vec
                            neu1e[j] += error * output_vec[j];
                            // update output vec
                            output_vec[j] += error * input_vec[j];
                        }
                    }
                    // use cblas
                    // TODO:
                }
            }
            inline float DotProduct(float *vec1, float *vec2) {
                float rs = 0.0f;
                for(int i = 0; i < n_dim; ++ i) {
                    rs += (vec1[i] * vec2[i]);
                }
                return rs;
            }
            inline void NegativeSample(std::vector<size_t> &idx_vec, size_t pos_index, std::uniform_real_distribution<float> &ud) {
                // binary search to find the insertion position in cumulative table
                while (static_cast<int>(idx_vec.size()) <= n_neg) {
                    float prob = ud(rng);
                    size_t idx;
                    size_t left = 0, right = cum_table.size() - 1;
                    // std::cout << "prob: " << prob << std::endl;
                    while(left <= right) {
                        idx = left + (right - left) / 2;
                        if(idx == 0 || idx == cum_table.size() - 1) break;
                        if(cum_table[idx] >= prob && cum_table[idx - 1] < prob) break;
                        else if(cum_table[idx] < prob && cum_table[idx + 1] >= prob) { idx++; break; }
                        else if(cum_table[idx - 1] >= prob) right = idx - 1;
                        else    left = idx + 1;
                    }
                    if(vocab_vec[idx].index != pos_index)   idx_vec.push_back(vocab_vec[idx].index);
                }
                // for(auto idx : idx_vec) {
                //     std::cout << idx << ",";
                // }
                // std::cout << std::endl;

            }
            inline bool LoadSentences(const char *filename, std::vector<std::vector<size_t> > &sentences) {
                std::ifstream fin(filename);
                if(fin.fail()) {
                    std::cerr << filename << " open failed." << std::endl;
                    return false;
                }
                std::string line, word;
                size_t idx;
                // rng.seed(static_cast<int>(time(NULL)));
                std::uniform_real_distribution<float> ud(0.0, 1.0);
                while(std::getline(fin, line)) {
                    std::istringstream ss(line);
                    std::vector<size_t> sentence;
                    while(!ss.eof()) {
                        if(!(ss >> word))   break;
                        assert(word2idx.size() > 0 && word2idx.size() == vocab_vec.size());
                        auto iter = word2idx.find(word);
                        if(iter == word2idx.end())  continue;
                        idx = iter->second;
                        float rand_prob = ud(rng);
                        if(Downsample(vocab_vec[idx], rand_prob)) continue;
                        sentence.push_back(idx);
                    }
                    sentences.push_back(sentence);
                }
                std::cout << "Load Train Sentences Complete." << std::endl;
                return true;
            }
            inline bool Downsample(const Vocab v, float rand_prob) {
                float threshold_cnt = sample * n_sum_cnt;
                float p = (std::sqrt(v.cnt / threshold_cnt) + 1) * (threshold_cnt / v.cnt);
                if (rand_prob > p)    return true;
                else    return false;
            }
            inline bool GenRawVocab(std::unordered_map<std::string, int> &raw_map) {
                std::ifstream fin(filename);
                if(fin.fail()) {
                    std::cerr << filename << " open failed." << std::endl;
                    return false;
                }
                std::string line, word;
                size_t n_word = 0, n_line = 0;
                while(std::getline(fin, line)) {
                    std::istringstream ss(line);
                    n_line ++;
                    while(!ss.eof()) {
                        if(!(ss >> word))   break;
                        n_word ++;
                        if(raw_map.find(word) == raw_map.end()) {
                            raw_map[word] = 0;
                        } else {
                            raw_map[word] += 1;
                        }
                    }
                }
                fin.close();
                std::cout << "Total " << n_line << " lines, " << n_word << " words, " << raw_map.size() << " unique words." << std::endl;
                return true;
            }
            inline void RefineVocab(std::unordered_map<std::string, int> &raw_map) {
                for(auto iter = raw_map.begin(); iter != raw_map.end(); ++ iter) {
                    if(iter->second >= min_cnt) vocab_vec.push_back(Vocab(iter->first, iter->second));
                }
                std::sort(vocab_vec.begin(), vocab_vec.end());
                for(size_t i = 0; i < vocab_vec.size(); ++ i) {
                    vocab_vec[i].index = i;
                    word2idx[vocab_vec[i].word] = i;
                    n_sum_cnt += vocab_vec[i].cnt;
                }
            }
            inline void MakeCumTable() {
                cum_table.reserve(vocab_vec.size());
                float Z = 0.0f, power = 0.75;
                for(auto v : vocab_vec) {
                    Z += std::pow(v.cnt, power);
                }
                float cumulative = 0.0f;
                for(auto v : vocab_vec) {
                    cumulative += std::pow(v.cnt, power) / Z;
                    cum_table.push_back(cumulative);
                }
                std::cout << "eps: " << 1.0 - cum_table[cum_table.size() - 1] << std::endl;
                // cum_table[cum_table.size() - 1] = 1.0;
            }
            inline void ResetWeight() {
                float *dptr0 = new float[vocab_vec.size() * n_dim];
                float *dptr1 = new float[vocab_vec.size() * n_dim];
                syn0.Set(dptr0, vocab_vec.size(), n_dim);
                syn1.Set(dptr1, vocab_vec.size(), n_dim);
                syn1.Reset();
                // init weight, uniform distribution, init syn0 only ???
                // rng.seed(static_cast<int>(time(NULL)));
                std::uniform_real_distribution<float> ud(-0.5, 0.5);
                for(int i = 0; i < n_dim; ++ i) {
                    for(size_t j = 0; j < vocab_vec.size(); ++ j) {
                        syn0[j][i] = ud(rng);
                    }
                }
            }
            inline void Save(const char *filename) {
                std::string file_syn0 = std::string(filename) + std::string("_syn0.bin");
                std::string file_syn1 = std::string(filename) + std::string("_syn1neg.bin");
                std::string file_vocab = std::string(filename) + std::string("_vocab.txt");
                syn0.Save(file_syn0.c_str());
                syn1.Save(file_syn1.c_str());
                std::ofstream fout(file_vocab.c_str());
                if(fout.fail()) {
                    std::cerr << file_vocab << " open fail." << std::endl;
                    return;
                }
                fout << n_dim << std::endl;
                for(auto v : vocab_vec) {
                    fout << v.word << " " << v.index << " " << v.cnt << std::endl;
                }
                fout.close();
            }
            inline std::vector<Vocab> GetVocab() const {
                return vocab_vec;
            }
            inline const Mat& GetSyn0() const {
                return syn0;
            }
            inline const Mat& GetSyn1() const {
                return syn1;
            }
        private:
            const char *filename;
            int n_dim;
            int n_window;
            int min_cnt;
            std::vector<Vocab> vocab_vec;
            std::unordered_map<std::string, size_t> word2idx;
            std::vector<float> cum_table;
            Mat syn0;
            Mat syn1;
            long n_sum_cnt;
            float sample;
            int sg;
            int n_neg;
            float lr;
            int n_iter;
            std::mt19937 rng;
    };
}

#endif /*WORD2VEC_H*/
