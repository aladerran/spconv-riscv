#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <tuple>
#include <vector>
#include "conv2d.h"
#include "helper.cpp"
// #include "systolic.cpp"
#include "include/gemmini_testutils.h"

using namespace std;



void testbench() {

    bool debug = false;

    int nIn = 3;
    int nOut = 4;
    bool use_bias = true;
    int dimension = 2;
    int filter_size = 2;
    std::vector<int> spatial_dimensions = {8, 8};


    Eigen::MatrixXf kernel = randomUniformMatrix(std::pow(filter_size, dimension), nOut * nIn, -10.0f, 10.0f);
    Eigen::VectorXf bias = randomUniformVector(nOut, -10.0f, 10.0f);

    std::cout << "kernel:\n" << kernel << std::endl;
    std::cout << "bias:\n" << bias << std::endl;


    auto input = createInput(nIn, spatial_dimensions, false, 1, false);

    auto batch_input = std::get<0>(input);
    auto batch_update_locations = std::get<1>(input);

    std::cout << "batch_input: \n" << batch_input << std::endl;
    // batch_update_locations needs revise


    auto sparse_conv = AsynSparseConvolution2Dcpp(dimension, nIn, nOut, filter_size, true, use_bias, debug);
    sparse_conv.setParameters(bias, kernel);


    uint64_t start = read_cycles();

    // sparse_conv_forward = sparse_conv.forward(batch_update_locations, feature_map=batch_input);

    uint64_t end = read_cycles();

    // std::cout<< "sparse conv took " << end-start << " cycles" << endl;
    
}


int main(){

    std::cout<< "Test begins..." << endl;
    testbench();

    return 0;

}