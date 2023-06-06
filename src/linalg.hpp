#pragma once
#include <cmath>
#include <iostream>
#include <vector>
namespace linalg {
void print_matrix_double(const std::vector<std::vector<double>> &mat);

void print_vector_double(const std::vector<double> &vec);

std::vector<double> matvec(const std::vector<std::vector<double>> &mat,
                           const std::vector<double> &vec);

std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>> &mat);

std::vector<std::vector<double>> matmat(
    const std::vector<std::vector<double>> &amat,
    const std::vector<std::vector<double>> &bmat);

std::vector<std::vector<double>> matinv(std::vector<std::vector<double>> mat);
}