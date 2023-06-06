#include <omp.h>

#include "gfunc.hpp"
#include "init.hpp"
#include "linalg.hpp"
#include "smc_fault.hpp"
#include "smc_slip.hpp"

int main() {
    omp_set_num_threads(80);
    // length of the fault [km]
    double lxi = 25;
    double leta = 25;
    // num of patch
    int nxi = 5;
    int neta = 5;

    // generate output dir
    std::string output_dir = "output/";
    std::string op = "mkdir -p " + output_dir;
    system(op.c_str());

    // set fault geometry
    // cny_fault[patch_id] = {node_id}
    std::vector<std::vector<int>> cny_fault;
    // coor_fault[node_id] = {node_coordinate}
    std::vector<std::vector<double>> coor_fault;
    // node_to_elem[node_id] = {patch_id containing the node}
    std::unordered_map<int, std::vector<int>> node_to_elem;
    // id_dof = {node_id which have degree of freedom}
    // slip value at node on the edge is fixed to be zero, no degree of freedom.
    std::vector<int> id_dof;
    init::discretize_fault(lxi, leta, nxi, neta, cny_fault, coor_fault,
                           node_to_elem, id_dof);

    // read observation data
    // coordinate of the observation points (x, y)
    std::vector<std::vector<double>> obs_points;
    // line-of-sight direction unit vector (e_x, e_y, e_z)
    std::vector<std::vector<double>> obs_unitvec;
    // error(sqrt(variance)) for observation
    std::vector<double> obs_sigma;
    // line-of-sight displacement
    std::vector<double> dvec;
    // number of observations for SAR/GNSS
    int nsar, ngnss;
    init::read_observation("input/observation_with_gnss_reduced.csv",
                           obs_points, obs_unitvec, obs_sigma, dvec, nsar,
                           ngnss);

    // calculate laplacian matrix
    const int nnode_fault = coor_fault.size();
    const double dxi = lxi / nxi;
    const double deta = leta / neta;
    // matrix L
    auto lmat = init::gen_laplacian(nnode_fault, nxi, neta, dxi, deta, id_dof);
    // matrix L^T L
    auto llmat = init::calc_ll(lmat);

    double xf = 1.;
    double yf = -19;
    double zf = -13;
    double strike = 2.;
    double dip = 72.;
    double log_sigma_sar2 = 0;
    double log_sigma_gnss2 = 1.;
    double log_alpha2 = -4;
    std::vector<double> particle = {
        xf, yf, zf, strike, dip, log_sigma_sar2, log_sigma_gnss2, log_alpha2};
    for (int iter = 0; iter < 1; iter++) {
        double tmp = smc_fault::calc_likelihood(
            particle, cny_fault, coor_fault, dvec, obs_points, obs_unitvec,
            obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss, lmat, llmat);
        std::cout << tmp << std::endl;
    }
    std::exit(1);
    // range for xf, yf, zf, strike, dip, log_sigma2_sar, log_sigma2_gnss,
    // log_alpha2
    std::vector<std::vector<double>> range = {{-10, 10}, {-30, 0}, {-30, -1},
                                              {-20, 20}, {50, 90}, {-2, 2},
                                              {-2, 2},   {-10, -2}};

    // sequential monte carlo sampling for fault parameters
    // numbers of samples for approximation of distributions
    int nparticle = 2000;
    std::vector<std::vector<double>> particles;
    smc_fault::smc_exec(particles, "output/", range, nparticle, cny_fault,
                        coor_fault, obs_points, dvec, obs_unitvec, obs_sigma,
                        leta, node_to_elem, id_dof, lmat, llmat, nsar, ngnss);
}
