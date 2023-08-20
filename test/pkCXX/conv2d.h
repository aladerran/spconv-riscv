#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include <Eigen/Core>
#include "src/async_sparse_conv2d.h"
#include "src/rulebook.h"


Eigen::MatrixXi convertToMatrix(const std::vector<Eigen::Vector2i>& vec) {
    Eigen::MatrixXi result(vec.size(), 2);
    for (size_t i = 0; i < vec.size(); ++i) {
        result.row(i) = vec[i].transpose();
    }
    return result;
}


class AsynSparseConvolution2Dcpp {
private:
    AsynSparseConvolution2D conv;
    int nIn, nOut, filter_size, dimension;
    bool first_layer;

public:
    AsynSparseConvolution2Dcpp(int dimension, int nIn, int nOut, int filter_size, bool first_layer=false, bool use_bias=false, bool debug=false)
        : conv(dimension, nIn, nOut, filter_size, first_layer, use_bias, debug),
          nIn(nIn), nOut(nOut), filter_size(filter_size), dimension(dimension), first_layer(first_layer) {}

    void setParameters(const Eigen::MatrixXf& weights, const Eigen::VectorXf& bias) {
        int reshapedRows = std::pow(filter_size, dimension);
        Eigen::MatrixXf reshaped_weights = Eigen::Map<const Eigen::MatrixXf>(weights.data(), reshapedRows, nIn * nOut);
        Eigen::VectorXf reshaped_bias = Eigen::Map<const Eigen::VectorXf>(bias.data(), bias.size()); 
        
        conv.setParameters(reshaped_bias, reshaped_weights);
    }

    using ReturnType = std::tuple<Eigen::MatrixXi, Eigen::MatrixXf, ActiveMatrix, RuleBook>;
    ReturnType forward(const std::vector<Eigen::Matrix<int, 2, 1>>& update_location,
                                    const Eigen::MatrixXf& feature_map,
                                    const Eigen::MatrixXf& active_sites_map,
                                    RuleBook& rule_book){
                                        
        int H = feature_map.rows();
        int W = feature_map.cols() / nIn; 

        bool no_updates = false;
        if (update_location.size() == 0) 
        {
            no_updates = true;
            std::vector<Eigen::Matrix<int, 2, 1>> update_location_dummy(1, Eigen::Matrix<int, 2, 1>::Zero());
        }

        Eigen::MatrixXf reshaped_feature_map = feature_map;
        reshaped_feature_map.resize(H * W, nIn);

        if (first_layer_) 
        {
            initMaps(H, W);
            auto active_sites_map = initActiveMap(reshaped_feature_map, update_location);
            RuleBook rule_book(H, W, filter_size_, dimension_);
        } 
        else 
        {
            initMaps(H, W);
        }

        std::vector<Eigen::Matrix<int, 2, 1>> new_update_locations;
        Eigen::MatrixXf output_map;
        Eigen::MatrixXf updated_active_sites_map;
        
        std::tie(new_update_locations, output_map, updated_active_sites_map) = forward(update_location, 
                                                                                    reshaped_feature_map, 
                                                                                    active_sites_map, 
                                                                                    rule_book, 
                                                                                    no_updates);

        output_map.resize(H, W * nOut); 
        updated_active_sites_map.resize(H, W); 

        return std::make_tuple(new_update_locations, output_map, updated_active_sites_map, rule_book);
    }   

};