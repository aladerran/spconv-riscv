#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include <Eigen/Core>
#include "src/rpg_asynet/async_sparse_conv2d.h"
#include "src/rpg_asynet/rulebook.h"


class AsynSparseConvolution2Dcpp {
private:
    AsynSparseConvolution2D conv;
    int nIn, nOut, filter_size, dimension;
    bool first_layer;

public:
    AsynSparseConvolution2Dcpp(int dimension, int nIn, int nOut, int filter_size, bool first_layer=false, bool use_bias=false, bool debug=false)
        : conv(dimension, nIn, nOut, filter_size, first_layer, use_bias, debug),
          nIn(nIn), nOut(nOut), filter_size(filter_size), dimension(dimension), first_layer(first_layer) {}

    void setParameters(const Eigen::VectorXf& bias, const Eigen::MatrixXf& weights) {
        conv.setParameters(bias, weights);
    }

    // forward() to be added...

};