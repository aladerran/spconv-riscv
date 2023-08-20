#include <iostream>
#include <Eigen/Dense>
#include <random>

Eigen::MatrixXf randomUniformMatrix(int rows, int cols, float low, float high) {
    Eigen::MatrixXf m(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(low, high);

    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            m(i, j) = dis(gen);
        }
    }

    return m;
}

Eigen::VectorXf randomUniformVector(int size, float low, float high) {
    Eigen::VectorXf v(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(low, high);

    for (int i = 0; i < size; i++) {
        v(i) = dis(gen);
    }

    return v;
}


std::tuple<Eigen::MatrixXf, std::vector<Eigen::Matrix<int, 2, 1>>, 
           std::vector<Eigen::MatrixXf>, std::vector<std::vector<Eigen::Matrix<int, 2, 1>>>>
createInput(int nIn = 1, std::vector<int> spatial_dimensions = {10, 20}, bool asynchronous_input = true, 
            int sequence_length = 3, bool simplified = false) {

    if (!asynchronous_input && sequence_length != 1) {
        throw std::invalid_argument("Expected the sequence length to be 1 for batch input. Got sequence length " 
                                     + std::to_string(sequence_length));
    }

    std::vector<Eigen::MatrixXf> asyn_input;
    std::vector<std::vector<Eigen::Matrix<int, 2, 1>>> asyn_update_locations;
    Eigen::MatrixXf batch_input = Eigen::MatrixXf::Zero(spatial_dimensions[0], spatial_dimensions[1] * nIn);

    int spatial_volume = spatial_dimensions[0] * spatial_dimensions[1];

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < sequence_length; i++) {
        int nr_cell_updates;
        Eigen::MatrixXf random_features;

        if (simplified) {
            nr_cell_updates = 1;
            std::uniform_int_distribution<> dis(-1, 1);
            random_features = Eigen::MatrixXf::Random(nr_cell_updates, nIn).unaryExpr([&](float val) { return static_cast<float>(dis(gen)); });
        } else {
            int high_threshold = std::min(spatial_volume / 2, 200);
            std::uniform_int_distribution<> dis1(1, std::max(high_threshold, 2));
            nr_cell_updates = dis1(gen);

            std::uniform_int_distribution<> dis2(-10, 10);
            random_features = Eigen::MatrixXf::Random(nr_cell_updates, nIn).unaryExpr([&](float val) { return static_cast<float>(dis2(gen)); });
        }

        // Exclude feature updates for difference of zero
        for (int j = 0; j < nr_cell_updates; j++) {
            if (random_features.row(j).squaredNorm() == 0) {
                random_features.row(j) = Eigen::MatrixXf::Ones(1, nIn);
            }
        }

        std::vector<int> random_permutation(spatial_volume);
        std::iota(random_permutation.begin(), random_permutation.end(), 0);
        std::shuffle(random_permutation.begin(), random_permutation.end(), gen);
        random_permutation.resize(nr_cell_updates);

        Eigen::MatrixXf asyn_input_i = Eigen::MatrixXf::Zero(spatial_volume, nIn);
        for (int j = 0; j < nr_cell_updates; j++) {
            asyn_input_i.row(random_permutation[j]) = random_features.row(j);
        }

        asyn_input_i.resize(spatial_dimensions[0], spatial_dimensions[1] * nIn);

        batch_input += asyn_input_i;
        asyn_input.push_back(batch_input);

        std::vector<Eigen::Matrix<int, 2, 1>> asyn_locations_i(nr_cell_updates);
        for (int j = 0; j < nr_cell_updates; j++) {
            asyn_locations_i[j] << random_permutation[j] / spatial_dimensions[1], 
                                    random_permutation[j] % spatial_dimensions[1];
        }

        asyn_update_locations.push_back(asyn_locations_i);
    }

    std::vector<Eigen::Matrix<int, 2, 1>> batch_update_locations;
    for (int i = 0; i < spatial_dimensions[0]; i++) {
        for (int j = 0; j < spatial_dimensions[1]; j++) {
            if (batch_input(i, j * nIn) != 0.0f) {
                batch_update_locations.push_back(Eigen::Matrix<int, 2, 1>(i, j));
            }
        }
    }

    if (asynchronous_input) {
        return {batch_input, batch_update_locations, asyn_input, asyn_update_locations};
    } else {
        return {batch_input, batch_update_locations, {}, {}};
    }
}

void print_batch_update_locations(const std::vector<Eigen::Matrix<int, 2, 1>>& vec) {
    std::cout << "batch update locations:" << std::endl;
    for(const auto& matrix : vec) {
        std::cout << matrix << std::endl;
    }
}