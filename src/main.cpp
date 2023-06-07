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
    // sparse matrix form of lmat
    std::vector<int> lmat_index;
    std::vector<double> lmat_val;
    init::gen_sparse_lmat(lmat, lmat_index, lmat_val);

    // Execute SMC for slip while fixing the fault
    std::vector<double> particle = {
        9.40751387e-01, -1.73475157e+01, -1.00443905e+01,
        4.96780337e-01, 7.16747382e+01,  2.06857834e-01,
        2.77420866e-01, -6.57536267e+00, 2.20408325e+02};
    double likelihood = smc_fault::calc_likelihood(
        particle, cny_fault, coor_fault, dvec, obs_points, obs_unitvec,
        obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss, lmat_index,
        lmat_val, llmat);
    std::cout << "result: " << likelihood << std::endl;
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
                        leta, node_to_elem, id_dof, lmat_index, lmat_val, llmat,
                        nsar, ngnss);
}
