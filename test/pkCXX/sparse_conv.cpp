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

    // Create Input
    int nIn = 5;
    int nOut = 8;
    bool use_bias = true;
    int dimension = 2;
    int filter_size = 3;
    std::vector<int> spatial_dimensions = {25, 25};

    Eigen::MatrixXf kernel = randomUniformMatrix(std::pow(filter_size, dimension), nOut * nIn, -10.0f, 10.0f);
    std::cout << "Kernel:\n" << kernel << std::endl;


    if (use_bias) {
        Eigen::VectorXf bias = randomUniformVector(nOut, -10.0f, 10.0f);
        std::cout << "Bias:\n" << bias << std::endl;
    }

    auto input = createInput(nIn, spatial_dimensions, false, 1, false);

    auto batch_input = std::get<0>(input);
    auto batch_update_locations = std::get<1>(input);

    std::cout << "batch_input: \n" << batch_input << std::endl;
    print_batch_update_locations(batch_update_locations);



    uint64_t start = read_cycles();




    uint64_t end = read_cycles();

    std::cout<< "sparse conv took " << end-start << " cycles" << endl;

}


int main(){

    std::cout<< "Test begins...\n" << endl;
    testbench();

    return 0;

}