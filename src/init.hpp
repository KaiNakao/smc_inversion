#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "linalg.hpp"
namespace init {
void discretize_fault(const double &lxi, const double &leta, const int &nxi,
                      const int &neta, std::vector<std::vector<int>> &cny,
                      std::vector<std::vector<double>> &coor,
                      std::unordered_map<int, std::vector<int>> &node_to_elem,
                      std::vector<int> &id_dof);

void read_observation(const std::string &path,
                      std::vector<std::vector<double>> &obs_points,
                      std::vector<std::vector<double>> &obs_unitvec,
                      std::vector<double> &obs_sigma, std::vector<double> &dvec,
                      int &nsar, int &ngnss);

std::vector<std::vector<double>> gen_laplacian(const int &nnode, const int &nxi,
                                               const int &neta,
                                               const double &dxi,
                                               const double &deta,
                                               const std::vector<int> &id_dof);

std::vector<std::vector<double>> calc_ll(
    const std::vector<std::vector<double>> &lmat);
}  // namespace init
