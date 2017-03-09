#ifndef LAYERS_H
#define LAYERS_H

#include "mat.h"
#include <cstdio>
#include <vector>
#include <limits>
#include <random>

class ConvLayer {     // nlp conv only
    public:
        ConvLayer() {}
        ConvLayer(size_t n_dim, std::vector<size_t> filter_width_vec, size_t n_feat_map)
            : n_dim(n_dim), filter_width_vec(filter_width_vec), n_feat_map(n_feat_map) {
            rng.seed(static_cast<int>(time(NULL)));
        }
        virtual ~ConvLayer() {
            for(size_t i = 0; i < filter_vec.size(); ++ i) {
                for(size_t j = 0; j < filter_vec[i].size(); ++ j) {
                    delete [] filter_vec[i][j].dptr;
                }
            }
            for(size_t i = 0; i < bias_vec.size(); ++ i) {
                delete [] bias_vec[i].dptr;
            }
        }
        inline size_t GetDim() const {
            return this->n_dim;
        }
        inline std::vector<size_t> GetFilterWidthVec() const {
            return this->filter_width_vec;
        }
        inline size_t GetFeatMapNum() const {
            return this->n_feat_map;
        }
        inline void LoadParams(FILE *fp) {
            filter_vec.resize(filter_width_vec.size());
            for(size_t i = 0; i < filter_vec.size(); ++ i) {
                filter_vec[i].resize(n_feat_map);
            }
            bias_vec.resize(filter_width_vec.size());
            for(size_t i = 0; i < filter_width_vec.size(); ++ i) {
                for(size_t j = 0; j < n_feat_map; ++ j) {
                    float *weights_ptr = new float[n_dim * filter_width_vec[i]];
                    fread(weights_ptr, 1, sizeof(float) * n_dim * filter_width_vec[i], fp);
                    filter_vec[i][j].Set(weights_ptr, filter_width_vec[i], n_dim);
                }
                float *bias_ptr = new float[n_feat_map];
                fread(bias_ptr, 1, sizeof(float) * n_feat_map, fp);
                bias_vec[i].Set(bias_ptr, 1, n_feat_map);
            }
        }
        inline void LoadParams(const char * filename) {
            FILE *fp = fopen(filename, "rb");
            if (fp == NULL) {
                std::cerr << filename << " open error.\n";
                return;
            } else {
                LoadParams(fp);
                fclose(fp);
            }           
        }
        inline void SaveParams(FILE *fp) {
            for(size_t i = 0; i < filter_width_vec.size(); ++ i) {
                for(size_t j = 0; j < n_feat_map; ++ j) {
                    fwrite(filter_vec[i][j].dptr, 1, sizeof(float) * n_dim * filter_width_vec[i], fp);
                }
                fwrite(bias_vec[i].dptr, 1, sizeof(float) * n_feat_map, fp);
            }
        }
        inline void SaveParams(const char * filename) {
            FILE *fp = fopen(filename, "wb");
            if (fp == NULL) {
                std::cerr << filename << " open error.\n";
                return;
            } else {
                SaveParams(fp);
                fclose(fp);
            }
        }
        inline void Init() {
            filter_vec.resize(filter_width_vec.size());
            for(size_t i = 0; i < filter_vec.size(); ++ i) {
                filter_vec[i].resize(n_feat_map);
            }
            bias_vec.resize(filter_width_vec.size());
            for(size_t i = 0; i < filter_width_vec.size(); ++ i) {
                float fan_in = filter_width_vec[i] * n_dim;
                float fan_out = n_feat_map * fan_in / (25 - filter_width_vec[i] + 1);
                float bound = std::sqrt( 6.0f / (fan_in + fan_out) );
                std::uniform_real_distribution<float> ud(- bound, bound);
                for(size_t j = 0; j < n_feat_map; ++ j) {
                    float *weights_ptr = new float[n_dim * filter_width_vec[i]];
                    for(size_t k = 0; k < n_dim * filter_width_vec[i]; ++ k) {
                        weights_ptr[k] = ud(rng);
                    }
                    filter_vec[i][j].Set(weights_ptr, filter_width_vec[i], n_dim);
                }
                float *bias_ptr = new float[n_feat_map];
                bias_vec[i].Set(bias_ptr, 1, n_feat_map);
                bias_vec[i].Reset();
            }
        }
        inline float Conv2d(const mat::Mat &patch, const mat::Mat &weights) {
            size_t n_total = patch.n_row * patch.n_col;
            // fprintf(stderr, "patch shape: %s\n", patch.Shape().c_str());
            // fprintf(stderr, "weights shape: %s\n", weights.Shape().c_str());
            assert(n_total == weights.n_row * weights.n_col);
            float result = 0.0f;
            for(size_t i = 0; i < n_total; ++ i) {
                result += patch.dptr[i] * weights.dptr[n_total - 1 - i];    // flip filter
            }
            return result;
        }
        inline void ForwardNoGemm(mat::Mat &input, mat::Mat &output) {
            std::vector<size_t> max_idx_vec;
            ForwardNoGemm(input, output, max_idx_vec);
        }
        inline void ForwardNoGemm(mat::Mat &input, mat::Mat &output, std::vector<size_t> &max_idx_vec) {
            // input already do padding
            // output had enough memory   
            for(size_t i = 0; i < filter_width_vec.size(); ++ i) {
                for(size_t j = 0; j < n_feat_map; ++ j) {
                    float max_num = -std::numeric_limits<float>::max();
                    size_t max_idx = 0;
                    for(size_t k = 0; k < input.n_row - filter_width_vec[i] + 1; ++ k) {  
                        mat::Mat patch = input.SubMat(k, k + filter_width_vec[i]);
                        // fprintf(stderr, "input.shape: %s\n", input.Shape().c_str());
                        float result = Conv2d(patch, filter_vec[i][j]);
                        // fprintf(stderr, "%f\t", result);
                        if (result > max_num) { 
                            max_num = result;
                            max_idx = k;
                        }
                    }
                    // fprintf(stderr, "\n");
                    output[i][j] = max_num;
                    max_idx_vec.push_back(max_idx);
                }
                // fprintf(stderr, "filter_width(%lu), output: %s\n", filter_width_vec[i], output.ToString().c_str());
                output.SubMat(i, i+1) += bias_vec[i];
                // fprintf(stderr, "weights : %s\n", filter_vec[i][9].ToString().c_str());
                // fprintf(stderr, "b (OK): %s\n", bias_vec[i].ToString().c_str());
                // fprintf(stderr, "output + b: %s\n", output.ToString().c_str());
            }
            output.Reshape(1, filter_width_vec.size() * n_feat_map);
        }
        inline void Forward(mat::Mat &input, mat::Mat &output) {
            // TODO
        }
        inline std::vector<std::vector<mat::Mat> > & GetFilterVec() {
            return filter_vec;
        }
        inline std::vector<mat::Mat> & GetBiasVec() {
            return bias_vec;
        }
    private:
        size_t n_dim;
        std::vector<size_t> filter_width_vec;
        size_t n_feat_map;
        std::vector<std::vector<mat::Mat> > filter_vec;
        std::vector<mat::Mat> bias_vec;
        std::mt19937 rng;
};
class FullyConnectedLayer {
    public:
        FullyConnectedLayer() {}
        FullyConnectedLayer(size_t n_row, size_t n_col) : n_row(n_row), n_col(n_col) {}
        virtual ~FullyConnectedLayer() {
            delete [] weights.dptr;
            delete [] bias.dptr;
        }
        inline void LoadParams(FILE *fp) {
            float *weights_ptr = new float[n_row * n_col];
            float *bias_ptr = new float[n_col];
            fread(weights_ptr, 1, sizeof(float) * n_row * n_col, fp);
            fread(bias_ptr, 1, sizeof(float) * n_col, fp);
            weights.Set(weights_ptr, n_row, n_col);
            bias.Set(bias_ptr, 1, n_col);
        }
        inline size_t GetInputDim() const {
            return this->n_row;
        }
        inline size_t GetOutputDim() const {
            return this->n_col;
        }
        inline void Forward(mat::Mat &input, mat::Mat &output) {
            // doesnot care input and output
            output.deepcopy(bias);
            mat::sgemm(input, weights, output);  // output = dot(input, weights) + bias;
        }
    private:
        size_t n_row;
        size_t n_col;
        mat::Mat weights;
        mat::Mat bias;
};

enum Activation {Sigmoid=0, Tanh, Relu};
static inline void sigmoid(mat::Mat &m) {
    size_t n_total = m.n_row * m.n_col;
    for(size_t i = 0; i < n_total; ++ i) {
        m.dptr[i] = 1.0f / ( 1.0 + exp(-m.dptr[i]) );
    }
}
static inline float sigmoid_derivative(float e) {
    return e * (1.0f - e);
}
static inline void sigmoid_derivative(mat::Mat &m) {
    size_t n_total = m.n_row * m.n_col;
    for(size_t i = 0; i < n_total; ++ i) {
        m.dptr[i] = sigmoid_derivative(m.dptr[i]);
    }
}
static inline void tanh_mat(mat::Mat &m) {
    size_t n_total = m.n_row * m.n_col;
    for(size_t i = 0; i < n_total; ++ i) {
        m.dptr[i] = tanh(m.dptr[i]);
    }
}
static inline float tanh_derivative(float e) {
    return 1.0f - e * e;
}
static inline void tanh_derivative(mat::Mat &m) {
    size_t n_total = m.n_row * m.n_col;
    for(size_t i = 0; i < n_total; ++ i) {
        m.dptr[i] = tanh_derivative(m.dptr[i]);
    }
}
static inline void relu(mat::Mat &m) {
    size_t n_total = m.n_row * m.n_col;
    for(size_t i = 0; i < n_total; ++ i) {
        m.dptr[i] = (m.dptr[i] > 0.0f ? m.dptr[i] : 0.0f);
    }
}
static inline void relu_derivative(mat::Mat &m) {
    size_t n_total = m.n_row * m.n_col;
    for(size_t i = 0; i < n_total; ++ i) {
        m.dptr[i] = (m.dptr[i] > 0.0f ? 1.0f : 0.0f);
    }
}
class ActivationLayer {
    public:
        ActivationLayer(Activation type) : type(type) {}
        inline void Forward(mat::Mat &m) {
            switch (type) {
                case Sigmoid:
                    sigmoid(m);
                    break;
                case Tanh:
                    tanh_mat(m);
                    break;
                case Relu:
                    relu(m);
                    break;
                default:
                    break;
            }
        }
        inline void Backward(mat::Mat &m) {
            switch (type) {
                case Sigmoid:
                    sigmoid_derivative(m);
                    break;
                case Tanh:
                    tanh_derivative(m);
                    break;
                case Relu:
                    relu_derivative(m);
                    break;
                default:
                    break;
            }
        }
    private:
        Activation type;
};

#endif /*LAYERS_H*/
