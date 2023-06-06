#pragma once
#include <mkl_lapacke.h>
#include <omp.h>

#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

#include "linalg.hpp"
namespace gfunc {
extern "C" {
void dc3d0_(float *ALPHA, float *X, float *Y, float *Z, float *DEPTH,
            float *DIP, float *POT1, float *POT2, float *POT3, float *POT4,
            float *UX, float *UY, float *UZ, float *UXX, float *UYX, float *UZX,
            float *UXY, float *UYY, float *UZY, float *UXZ, float *UYZ,
            float *UZZ, int *IRET);

void dc3d_(float *ALPHA, float *X, float *Y, float *Z, float *DEPTH, float *DIP,
           float *AL1, float *AL2, float *AW1, float *AW2, float *DISL1,
           float *DISL2, float *DISL3, float *UX, float *UY, float *UZ,
           float *UXX, float *UYX, float *UZX, float *UXY, float *UYY,
           float *UZY, float *UXZ, float *UYZ, float *UZZ, int *IRET);
}

std::vector<std::vector<double>> gen_unit_slip(
    const std::vector<std::vector<double>> &coor, const int &inode,
    const int &idirection);

void call_dc3d0(const double &xsource, const double &ysource,
                const double &zsource, const double &xobs, const double &yobs,
                const double &uxi, const double &ueta, float &dip,
                const double &area, const double &strike,
                std::vector<double> &uret);

std::vector<double> calc_responce(
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &slip_dist, const double &xobs,
    const double &yobs, const double &leta, const double &xf, const double &yf,
    const double &zf, const double &strike, const double dip,
    const std::vector<int> &target_ie);

std::vector<std::vector<double>> calc_responce_dist(
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &slip_dist, const double &leta,
    const double &xf, const double &yf, const double &zf, const double &strike,
    const double dip, const std::vector<int> &target_ie, const int &nsar,
    const int &ngnss);

std::vector<std::vector<double>> calc_greens_func(
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<std::vector<double>> &obs_unitvec, const double &leta,
    const double &xf, const double &yf, const double &zf, const double &strike,
    const double &dip,
    const std::unordered_map<int, std::vector<int>> &node_to_elem,
    const std::vector<int> &id_dof, const int &nsar, const int &ngnss);
}  // namespace gfunc
