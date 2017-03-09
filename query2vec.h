#ifndef QUERY_EMBEDDING_H
#define QUERY_EMBEDDING_H

#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <utility>
#include <omp.h>
#include "mat.h"
#include "layers.h"
#include "MurmurHash3.h"

using mat::Mat;
struct Vocab {
    std::string word;
    int cnt;
    size_t index;
    std::vector<size_t> term_vec;
    std::vector<size_t> ngram_vec;
    Vocab(std::string _word, int _cnt) : word(_word), cnt(_cnt), index(0) {}
    Vocab(std::string _word, int _cnt, const std::vector<size_t> &_term_vec, const std::vector<size_t> &_ngram_vec) 
        : word(_word), cnt(_cnt), index(0), term_vec(_term_vec), ngram_vec(_ngram_vec) {}
    bool operator<(const Vocab& rhs) const {
        return rhs.cnt < cnt;
    }
};
const static int THREAD_NUM = omp_get_max_threads();
// const static int THREAD_NUM = 1;
class Query2Vec {
    public:
        Query2Vec() {}
        Query2Vec(const char *filename, const char *seg_filename, int n_dim, int n_window, int min_cnt, float sample, int sg, int n_neg, float lr, int n_iter)
            : filename(filename), seg_filename(seg_filename), n_dim(n_dim), n_window(n_window), min_cnt(min_cnt),
              n_sum_cnt(0), sample(sample), sg(sg), n_neg(n_neg), lr(lr), n_iter(n_iter) {
                // rng.seed(static_cast<int>(time(NULL)));
                rng.seed(1234);
                max_term_idx = 0;
                n_ngram = 0;
                size_t filter_widths[] = {1, 2, 3};
                std::vector<size_t> filter_width_vec(filter_widths, filter_widths + 3);
                size_t n_feat_map = 20;
                conv_layer = new ConvLayer(n_dim, filter_width_vec, n_feat_map);
                activation_layer = new ActivationLayer(Sigmoid);
                conv_layer->Init();
                is_hash = true;
                n_bucket = 1000000;
                total_loss = 0.0f;
              } 
        virtual ~Query2Vec() {
            delete [] syn0.dptr;
            syn0.dptr = NULL;
            delete [] syn1.dptr;
            syn1.dptr = NULL;
            delete [] syn0_term.dptr;
            syn0_term.dptr = NULL;
            delete [] syn0_ngram.dptr;
            syn0_ngram.dptr = NULL;
            delete conv_layer;
            conv_layer = NULL;
            delete activation_layer;
            activation_layer = NULL;
        }
        inline bool BuildVocab() {
            std::ios::sync_with_stdio(false);
            std::unordered_map<std::string, std::pair<std::vector<size_t>, std::vector<size_t> > > seg_map;
            if (!LoadSegMap(seg_map)) { return false; }
            std::unordered_map<std::string, int> raw_map;
            if (!GenRawVocab(raw_map)) { return false; }
            RefineVocab(raw_map, seg_map);
            MakeCumTable(); // for negative sampling
            ResetWeight();
            std::cout << "Vocabulary Build Complete." << std::endl;
            return true;
        }
        inline void Train() {
            std::vector<std::vector<size_t> > sentences;
            LoadSentences(filename, sentences);
            // rng.seed(static_cast<int>(time(NULL)));
            std::cout << "thread num: " << THREAD_NUM << std::endl;
            std::uniform_int_distribution<int> ud(0, n_window - 1);
            for(int iter = 0; iter < n_iter; ++ iter) {
                size_t width = sentences.size() / 1000;
                size_t progress = 0;
                # pragma omp parallel for num_threads(THREAD_NUM)
                for(int i = 0; i < static_cast<int>(sentences.size()); ++ i) {
                    # pragma omp critical 
                    {
                        progress ++;
                        if(progress % width == 0) {
                            if (lr > 0.0001)    lr *= 0.999;
                            std::cout << "[PROGRESS";
                            std::cout << "] ##### " << (float)(progress) / (sentences.size() * n_iter) * 100.0 
                                      << " % ### lr: " << lr 
                                      << " ### loss: " << (total_loss / total_cnt) << "\r" << std::flush;
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
                        if(i != j)  TrainSgPairNgram(sentence[i], sentence[j]);
                        // if(i != j)  TrainSgPairCNN(sentence[i], sentence[j]);
                        // TrainSgPair(sentence[i], sentence[j]);
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
                float total_error = 0;
                for(int i = 0; i <= n_neg; ++ i) {
                    if (i == 0) {
                        label = 1.0f;
                    } else {
                        label = 0.0f;
                    }
                    output_vec = syn1[idx_vec[i]];
                    // output_vec = syn0[idx_vec[i]];
                    pred_label = 1.0 / ( 1 + exp( -DotProduct( input_vec, output_vec ) ) );
                    total_error += (label - pred_label);
                    error = (label - pred_label) * lr;
                    for(int j = 0; j < n_dim; ++ j) {
                        // save grad for input vec
                        neu1e[j] += error * output_vec[j];
                        // update output vec
                        output_vec[j] += error * input_vec[j];
                    }
                }
                # pragma omp critical
                {
                    total_loss += (total_error / (n_neg+1));
                    total_cnt ++ ;
                }
                for(int i = 0; i < n_dim; ++ i) {
                    input_vec[i] += neu1e[i];
                }
            }
        }
        inline bool IsQuery(size_t idx) {
            return !vocab_vec[idx].term_vec.empty();
        }
        inline void DoPadding(std::vector<size_t> &terms, size_t n) {
            std::vector<size_t> pad_idx_vec;
            for(size_t i = 0; i < n; ++ i)  pad_idx_vec.push_back(max_term_idx + 1);
            terms.insert(terms.begin(), pad_idx_vec.begin(), pad_idx_vec.end());
            terms.insert(terms.end(), pad_idx_vec.begin(), pad_idx_vec.end());
        }
        inline void TrainSgPairCNN(size_t input_index, size_t pos_index) {
            if(n_neg > 0) {
                // forward && backward
                // std::cout << vocab_vec[input_index].word << ", " << vocab_vec[input_index].term_vec.size() << std::endl;
                std::vector<size_t> term_vec = vocab_vec[input_index].term_vec;
                DoPadding(term_vec, 3);
                // std::cout << "forward input vec" << std::endl;
                std::vector<size_t> max_idx_vec;
                Mat terms_mat(NULL, term_vec.size(), n_dim);
                Mat conv_output = syn0.SubMat(input_index, input_index + 1);
                // std::cout << "-----------------------------------\n";
                // std::cout << "input_index: " << input_index << ", IsQuery(input_index): " << IsQuery(input_index) << std::endl;
                std::vector<size_t> filter_width_vec = conv_layer->GetFilterWidthVec();
                size_t n_feat_map = conv_layer->GetFeatMapNum();
                float tmp_arr0[60 * 60] = {0};
                float tmp_arr1[60 * 60] = {0};
                if (IsQuery(input_index)) {
                    // 1. construct terms mat
                    // terms_mat.dptr = new float[terms_mat.n_row * terms_mat.n_col];
                    terms_mat.dptr = tmp_arr0;
                    for (size_t i = 0; i < term_vec.size(); ++ i) {
                        terms_mat.SubMat(i, i+1).deepcopy(syn0_term.SubMat(term_vec[i], term_vec[i]+1));
                    }
                    // std::cout << terms_mat.ToString() << std::endl;
                    // 2. conv forward
                    conv_output.Reshape(filter_width_vec.size(), n_feat_map);
                    # pragma omp critical
                    {
                    conv_layer->ForwardNoGemm(terms_mat, conv_output, max_idx_vec);
                    activation_layer->Forward(conv_output);
                    }
                    // std::cout << "conv_output:\n" << conv_output.ToString() << std::endl;
                }
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
                float total_error = 0;
                std::vector<size_t> bad_idx_vec;
                for(int i = 0; i <= n_neg; ++ i) {
                    if (i == 0) {
                        label = 1.0f;
                    } else {
                        label = 0.0f;
                    }
                    // std::cout << "forward output" << std::endl;
                    output_vec = syn1[idx_vec[i]];
                    pred_label = 1.0 / ( 1 + exp( -DotProduct( input_vec, output_vec ) ) );
                    // std::cout << "backward output" << std::endl;
                    total_error += (label - pred_label);
                    error = (label - pred_label) * lr;
                    for(int j = 0; j < n_dim; ++ j) {
                        // save grad for input vec
                        neu1e[j] += error * output_vec[j];
                        // update output vec
                        output_vec[j] += error * input_vec[j];
                    }
                }
                if (isnan(total_error)) {
                    std::cout << "total error: " << total_error / (n_neg+1) << std::endl;
                    exit(-1);
                }
                # pragma omp critical
                {
                    total_loss += (total_error / (n_neg+1));
                    total_cnt ++ ;
                }
                if (IsQuery(input_index)) {
                    Mat grad_terms(NULL, term_vec.size(), n_dim);
                    // grad_terms.dptr = new float[term_vec.size() * n_dim];
                    grad_terms.dptr = tmp_arr1;
                    grad_terms.Reset();
                    std::vector<std::vector<mat::Mat> > & filter_vec = conv_layer->GetFilterVec();
                    std::vector<mat::Mat> & bias_vec = conv_layer->GetBiasVec();
                    # pragma omp critical
                    {
                    for(int i = 0; i < n_dim; ++ i) {
                        float tmp_grad = neu1e[i] * sigmoid_derivative(input_vec[i]);
                        size_t max_idx = max_idx_vec[i];
                        mat::Mat patch_grad = grad_terms.SubMat(max_idx, max_idx + filter_width_vec[i / n_feat_map]);
                        mat::Mat patch = terms_mat.SubMat(max_idx, max_idx + filter_width_vec[i / n_feat_map]);
                        size_t n_total = patch_grad.n_row * patch_grad.n_col;
                        for(size_t ii = 0; ii < n_total; ++ ii) {
                            float tmp_patch_grad = 0.0f;
                            tmp_patch_grad = filter_vec[i / n_feat_map][i % n_feat_map].dptr[n_total - 1 - ii] * tmp_grad;
                            patch_grad.dptr[ii] += tmp_patch_grad;
                            // conv backward
                            filter_vec[i / n_feat_map][i % n_feat_map].dptr[ii] += (patch.dptr[n_total - 1 - ii] * tmp_grad);
                            bias_vec[i / n_feat_map].dptr[i % n_feat_map] += tmp_grad;
                        }
                    }
                    }
                    // update term embeddings with grad_terms
                    for(size_t i = 0; i < term_vec.size(); ++ i) {
                        syn0_term.SubMat(term_vec[i], term_vec[i]+1) += grad_terms.SubMat(i, i+1);
                    }
                    // delete new memory
                    // delete [] grad_terms.dptr;
                    // grad_terms.dptr = NULL;
                    // delete [] terms_mat.dptr;
                    // terms_mat.dptr = NULL; 
                } else {
                    for(int i = 0; i < n_dim; ++ i) {
                        input_vec[i] += neu1e[i];
                    }
                }

           }
        }
        inline void TrainSgPairNgram(size_t input_index, size_t pos_index) {
            if(n_neg > 0) {
                // forward && backward
                // std::cout << vocab_vec[input_index].word << ", " << vocab_vec[input_index].term_vec.size() << std::endl;
                std::vector<size_t> & term_vec = vocab_vec[input_index].term_vec;
                std::vector<size_t> & ngram_vec = vocab_vec[input_index].ngram_vec;
                Mat conv_output = syn0.SubMat(input_index, input_index + 1);
                conv_output.Reset();
                if (IsQuery(input_index)) {
                    // avg pooling
                    for(int i = 0; i < n_dim; ++ i) {
                        for(size_t j = 0; j < term_vec.size(); ++ j) {
                            conv_output[0][i] += syn0_term.GetRow(term_vec[j])[i];
                        }
                        // add ngrams
                        for(size_t j = 0; j < ngram_vec.size(); ++ j) {
                            conv_output[0][i] += syn0_ngram.GetRow(ngram_vec[j])[i];
                        }
                        conv_output[0][i] /= (term_vec.size() + ngram_vec.size());
                    }
                }
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
                float total_error = 0;
                for(int i = 0; i <= n_neg; ++ i) {
                    if (i == 0) {
                        label = 1.0f;
                    } else {
                        label = 0.0f;
                    }
                    output_vec = syn1[idx_vec[i]];
                    pred_label = 1.0 / ( 1 + exp( -DotProduct( input_vec, output_vec ) ) );
                    total_error += (label - pred_label);
                    error = (label - pred_label) * lr;
                    for(int j = 0; j < n_dim; ++ j) {
                        // save grad for input vec
                        neu1e[j] += error * output_vec[j];
                        // update output vec
                        output_vec[j] += error * input_vec[j];
                    }
                }
                if (isnan(total_error)) {
                    std::cout << "total error: " << total_error / (n_neg+1) << std::endl;
                    exit(-1);
                }
                # pragma omp critical
                {
                    total_loss += (total_error / (n_neg+1));
                    total_cnt ++ ;
                    // std::cout << "total error: " << total_error / (n_neg+1) << std::endl;
                }
                if (IsQuery(input_index)) {
                    // update term embeddings
                    for(int i = 0; i < n_dim; ++ i) {
                        float grad = neu1e[i] / (term_vec.size() + ngram_vec.size());
                        for(size_t j = 0; j < term_vec.size(); ++ j) {
                            syn0_term.GetRow(term_vec[j])[i] += grad;
                        }
                        for(size_t j = 0; j < ngram_vec.size(); ++ j) {
                            syn0_ngram.GetRow(ngram_vec[j])[i] += grad;    
                        }
                    }
                } else {
                    for(int i = 0; i < n_dim; ++ i) {
                        input_vec[i] += neu1e[i];
                    }
                }
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
                if (n_line % 100000 == 0)   std::cout << n_line << "\r" << std::flush;
            }
            fin.close();
            std::cout << "\nTotal " << n_line << " lines, " << n_word << " words, " << raw_map.size() << " unique words." << std::endl;
            return true;
        }
        inline bool LoadSegMap(std::unordered_map<std::string, std::pair<std::vector<size_t>, std::vector<size_t> > > &seg_map) {
            std::ifstream fin(seg_filename);
            if(fin.fail()) {
                std::cerr << seg_filename << " open failed." << std::endl;
                return false;
            }
            std::string line, word;
            size_t n_line = 0;
            while(std::getline(fin, line)) {
                n_line ++;
                std::vector<size_t> term_idx_vec;
                std::vector<size_t> ngram_idx_vec;
                ParseSegInfo(line, word, term_idx_vec, ngram_idx_vec);
                seg_map[word] = std::make_pair(term_idx_vec, ngram_idx_vec);    
            }
            fin.close();
            std::cout << "Load Seg Map Complete, Total " << n_line << " lines." << std::endl;
            return true;               
        }
        inline void ParseSegInfo(const std::string &line, std::string &word, std::vector<size_t> &term_idx_vec, std::vector<std::size_t> &ngram_idx_vec) {
            auto left_iter = line.begin(), right_iter = line.begin();
            size_t col_idx = 0;
            std::string seg_idx_rs, seg_rs;
            while(left_iter != line.end() && right_iter < line.end()) {
                if (*right_iter == '\t') {
                    if (col_idx == 1) {
                        word = std::string(left_iter, right_iter);
                    }
                    if (col_idx == 2) {
                        seg_rs = std::string(left_iter, right_iter);
                        // std::cout << "seg rs: " << seg_rs << std::endl;
                    }
                    right_iter ++;
                    left_iter = right_iter;
                    col_idx ++;
                }
                right_iter ++;
            }
            seg_idx_rs = std::string(left_iter, line.end());
            std::istringstream ss(seg_idx_rs);
            size_t term_idx;
            while(!ss.eof()) {
                ss >> term_idx;
                if (term_idx > max_term_idx)    max_term_idx = term_idx;
                term_idx_vec.push_back(term_idx);
            }
            std::vector<std::string> ngram_vec;
            GetNGrams(seg_rs, ngram_vec, 2);
            for(auto ngram : ngram_vec) {
                auto iter = ngram2idx.find(ngram);
                if (iter == ngram2idx.end()) {
                    if (!is_hash) {
                        ngram2idx[ngram] = n_ngram;
                        n_ngram ++;
                    } else {
                        size_t bucket = Murmurhash(ngram);
                        ngram2idx[ngram] = bucket % n_bucket;
                    }
                }
                ngram_idx_vec.push_back(ngram2idx[ngram]);
                // std::cout << ngram2idx[ngram]  << "/";
            }
            // std::cout << std::endl;
        }
        inline size_t Murmurhash(const std::string &ngram) {
            size_t val = 0;
            MurmurHash3_x64_128((void*)(ngram.c_str()), 32, 0, (void*)(&val));
            return val;
        }
        inline void GetNGrams(const std::string & seg_rs, std::vector<std::string> &ngram_vec, int n) {
            std::vector<std::string::const_iterator> iter_vec;
            iter_vec.push_back(seg_rs.begin());
            for(auto iter = seg_rs.begin(); iter != seg_rs.end(); ++ iter) {
                if (*iter == ' ' && (iter+1) != seg_rs.end())  iter_vec.push_back(iter + 1);
            }
            iter_vec.push_back(seg_rs.end());
            std::string ngram;
            for(int i = 2; i <= n; ++ i) {
                for(int j = 0; j < static_cast<int>(iter_vec.size()) - i; ++ j) {
                    ngram = std::string(iter_vec[j], iter_vec[j+i]);
                    ngram_vec.push_back(ngram);
                }
            }
            // end grams
            if (iter_vec.size() > 2)    ngram_vec.push_back(std::string(iter_vec[iter_vec.size()-2], seg_rs.end()) + " #");
            if (iter_vec.size() > 3)    ngram_vec.push_back(std::string(iter_vec[iter_vec.size()-3], seg_rs.end()) + " #");
        }
        inline void RefineVocab(const std::unordered_map<std::string, int> &raw_map, const std::unordered_map<std::string, std::pair<std::vector<size_t>, std::vector<size_t> > > &seg_map) {
            for(auto iter = raw_map.begin(); iter != raw_map.end(); ++ iter) {
                auto seg_iter = seg_map.find(iter->first);
                if (iter->second >= min_cnt) {
                    if (seg_iter == seg_map.end())  vocab_vec.push_back(Vocab(iter->first, iter->second));
                    else                            vocab_vec.push_back(Vocab(iter->first, iter->second, seg_iter->second.first, seg_iter->second.second));
                }
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
            float *dptr2 = new float[(max_term_idx + 2) * n_dim];
            syn0.Set(dptr0, vocab_vec.size(), n_dim);
            syn1.Set(dptr1, vocab_vec.size(), n_dim);
            syn1.Reset();
            if (is_hash)    n_ngram = n_bucket;
            float *dptr3 = new float[n_ngram * n_dim];
            syn0_term.Set(dptr2, max_term_idx + 2, n_dim);
            syn0_ngram.Set(dptr3, n_ngram, n_dim);
            // init weight, uniform distribution, init syn0 only ???
            // rng.seed(static_cast<int>(time(NULL)));
            std::uniform_real_distribution<float> ud(-0.5, 0.5);
            for(int i = 0; i < n_dim; ++ i) {
                for(size_t j = 0; j < vocab_vec.size(); ++ j) {
                    syn0[j][i] = ud(rng) / n_dim;
                }
                for(size_t j = 0; j < max_term_idx + 2; ++ j) {
                    syn0_term[j][i] = ud(rng) / n_dim;
                }
                for(size_t j = 0; j < n_ngram; ++ j) {
                    syn0_ngram[j][i] = ud(rng) / n_dim;
                }
            }
        }
        inline void Save(const char *filename) {
            std::string file_syn0 = std::string(filename) + std::string("_syn0.bin");
            std::string file_syn1 = std::string(filename) + std::string("_syn1neg.bin");
            std::string file_vocab = std::string(filename) + std::string("_vocab.txt");
            std::string file_syn0_term = std::string(filename) + std::string("_syn0_term.bin");
            std::string file_syn0_ngram = std::string(filename) + std::string("_syn0_ngram.bin");
            std::string file_ngram = std::string(filename) + std::string("_ngram.txt");
            syn0.Save(file_syn0.c_str());
            syn1.Save(file_syn1.c_str());
            syn0_term.Save(file_syn0_term.c_str());
            syn0_ngram.Save(file_syn0_ngram.c_str());
            std::ofstream fout(file_vocab.c_str()), fout_ngram(file_ngram.c_str());
            if(fout.fail()) {
                std::cerr << file_vocab << " open fail." << std::endl;
                return;
            }
            fout << n_dim << std::endl;
            for(auto v : vocab_vec) {
                fout << v.word << "\t" << v.index << "\t" << v.cnt << std::endl;
            }
            fout.close();
            if(fout_ngram.fail()) {
                std::cerr << file_ngram << " open fail." << std::endl;
                return;
            }
            for(auto gram : ngram2idx) {
                fout_ngram << gram.first << "\t" << gram.second << std::endl;
            }
            fout_ngram.close();
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
        const char *seg_filename;
        int n_dim;
        int n_window;
        int min_cnt;
        std::vector<Vocab> vocab_vec;
        std::unordered_map<std::string, size_t> word2idx;
        std::vector<float> cum_table;
        Mat syn0;
        Mat syn1;
        Mat syn0_term;
        long n_sum_cnt;
        float sample;
        int sg;
        int n_neg;
        float lr;
        int n_iter;
        std::mt19937 rng;
        size_t max_term_idx;
        ConvLayer *conv_layer;
        ActivationLayer *activation_layer;
        std::unordered_map<std::string, size_t> ngram2idx;
        size_t n_ngram;
        Mat syn0_ngram;
        bool is_hash;
        size_t n_bucket;
        float total_loss;
        size_t total_cnt;

        Query2Vec(const Query2Vec &other) {}
        Query2Vec& operator=(const Query2Vec &other);
};

#endif /*QUERY_EMBEDDING_H*/
