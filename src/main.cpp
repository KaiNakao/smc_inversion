#include <mpi.h>
#include <omp.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "gfunc.hpp"
#include "init.hpp"
#include "linalg.hpp"
#include "smc_fault.hpp"
#include "smc_slip.hpp"

int main(int argc, char *argv[]) {
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // length of the fault [km]
    // int nxi = lxi / 2.5;
    // int neta = leta / 1.5;
    int nxi = 6;
    int neta = 6;

    // generate output dir
    std::string output_dir = "test/";
    // std::string output_dir = "output_cvtest/";
    std::string op = "mkdir -p " + output_dir;
    system(op.c_str());
    op = "mkdir -p " + output_dir + "slip/";
    system(op.c_str());
    const int nparticle_slip = 1000;
    const int nparticle_fault = 1000;

    // // set fault geometry
    // // cny_fault[patch_id] = {node_id}
    // std::vector<std::vector<int>> cny_fault;
    // // coor_fault[node_id] = {node_coordinate}
    // std::vector<std::vector<double>> coor_fault;
    // // node_to_elem[node_id] = {patch_id containing the node}
    // std::unordered_map<int, std::vector<int>> node_to_elem;
    // // id_dof = {node_id which have degree of freedom}
    // // slip value at node on the edge is fixed to be zero, no degree of
    // // freedom.
    // std::vector<int> id_dof;
    // init::discretize_fault(lxi, leta, nxi, neta, cny_fault, coor_fault,
    //                        node_to_elem, id_dof);
    // // linalg::print_matrix_double(coor_fault);
    // for (int i = 0; i < cny_fault.size(); i++) {
    //     for (int v : cny_fault.at(i)) {
    //         std::cout << v << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::exit(1);

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

    // constrain max value for slip
    double max_slip = 3.;

    //     const int ndim = 8;
    //     const int nfold = 4;
    //     cross_validation(output_dir, nfold, nparticle_fault, cny_fault,
    //     coor_fault,
    //                      leta, node_to_elem, id_dof, lmat_index, lmat_val,
    //                      ndim, llmat, nparticle_slip, max_slip, myid,
    //                      numprocs);

    // // calculate diplacement for fixed fault and slip distribution
    // std::vector<double> particle = {0.96512816,  -14.13392699, -13.38230125,
    //                                 2.08155794,  74.50310946,  0.16392912,
    //                                 -0.16778241, -3.08278596};
    // std::string slip_path = "visualize/slip_mean.csv";
    // std::ifstream ifs(slip_path);
    // std::vector<double> slip(2 * coor_fault.size());
    // for (int inode = 0; inode < coor_fault.size(); inode++) {
    //     double sxi, seta;
    //     ifs >> sxi >> seta;
    //     // std::cout << sxi << " " << seta << std::endl;
    //     slip.at(2 * inode) = sxi;
    //     slip.at(2 * inode + 1) = seta;
    // }
    // std::vector<double> svec(2 * id_dof.size());
    // for (int idim = 0; idim < id_dof.size(); idim++) {
    //     int inode = id_dof.at(idim);
    //     svec.at(2 * idim) = slip.at(2 * inode);
    //     svec.at(2 * idim + 1) = slip.at(2 * inode + 1);
    // }

    // double xf = particle.at(0);
    // double yf = particle.at(1);
    // double zf = particle.at(2);
    // double strike = particle.at(3);
    // double dip = particle.at(4);
    // double log_sigma_sar2 = particle.at(5);
    // double log_sigma_gnss2 = particle.at(6);
    // double log_alpha2 = particle.at(7);
    // // Calculate greens function for the sampled fault
    // auto gmat = gfunc::calc_greens_func(cny_fault, coor_fault, obs_points,
    //                                     obs_unitvec, leta, xf, yf, zf,
    //                                     strike, dip, node_to_elem, id_dof,
    //                                     nsar, ngnss);
    // const int ndim = gmat.at(0).size();
    // std::vector<double> gmat_flat(gmat.size() * gmat.at(0).size());
    // for (int i = 0; i < gmat.size(); i++) {
    //     for (int j = 0; j < gmat.at(0).size(); j++) {
    //         gmat_flat.at(i * ndim + j) = gmat.at(i).at(j);
    //     }
    // }

    // std::vector<double> gsvec(dvec.size());
    // cblas_dgemv(CblasRowMajor, CblasNoTrans, dvec.size(), svec.size(), 1.,
    //             &gmat_flat[0], svec.size(), &svec[0], 1, 0., &gsvec[0], 1);
    // // for (int iobs = 0; iobs < dvec.size(); iobs++) {
    // //     std::cout << gsvec.at(iobs) << " " << dvec.at(iobs) << std::endl;
    // // }
    // std::ofstream ofs("visualize/dvec_est.dat");
    // for (int iobs = 0; iobs < dvec.size(); iobs++) {
    //     ofs << gsvec.at(iobs) << std::endl;
    // }
    // std::exit(1);

    // std::vector<double> particle = {
    //     0.96512816, -14.13392699, -13.38230125, 2.08155794, 74.50310946,
    //     0.16392912, -0.16778241,  -3.08278596,  10.,        10.,
    // };
    // double st_time, en_time;
    // st_time = MPI_Wtime();
    // double likelihood = smc_fault::calc_likelihood(
    //     particle, dvec, obs_points, obs_unitvec, obs_sigma, nsar, ngnss,
    //     nparticle_slip, max_slip, nxi, neta);
    // en_time = MPI_Wtime();
    // std::cout << " result: " << likelihood << std::endl;
    // std::cout << " etime: " << en_time - st_time << std::endl;
    // std::exit(0);

    // std::exit(0);

    // range for xf, yf, zf, strike, dip, log_sigma2_sar,
    // log_sigma2_gnss, log_alpha2
    std::vector<std::vector<double>> range = {
        {-10, 10}, {-30, 0}, {-30, -1}, {-20, 20}, {50, 90},
        {-2, 2},   {-2, 2},  {-10, 2},  {1, 50},   {1, 50}};
    std::vector<double> particles_flat;
    smc_fault::smc_exec(particles_flat, output_dir, range, nparticle_fault,
                        obs_points, dvec, obs_unitvec, obs_sigma, nsar, ngnss,
                        nparticle_slip, max_slip, nxi, neta, myid, numprocs);

    MPI_Finalize();
}
