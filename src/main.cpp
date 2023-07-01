#include <mpi.h>
#include <omp.h>

#include <fstream>

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

    std::string lxi_str = argv[1];
    std::string leta_str = argv[2];

    // length of the fault [km]
    double lxi = std::stod(lxi_str);
    double leta = std::stod(leta_str);
    // num of patch
    int nxi = lxi / 5;
    int neta = leta / 5;

    // generate output dir
    // std::string output_dir = "output_" + lxi_str + "_" + leta_str + "/all/";
    std::string output_dir = "output_mpi80000/";
    std::string op = "mkdir -p " + output_dir;
    system(op.c_str());
    const int nparticle_slip = 20000;
    const int nparticle_fault = 80000;
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
    // linalg::print_matrix_double(coor_fault);
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

    // std::ofstream ofs(output_dir + "cv_score.dat");
    // int nfold = 5;
    // for (int cv_id = 0; cv_id < nfold; cv_id++) {
    //     std::vector<std::vector<double>> obs_points;
    //     std::vector<std::vector<double>> obs_unitvec;
    //     std::vector<double> obs_sigma;
    //     std::vector<double> dvec;
    //     int nsar, ngnss;
    //     init::read_observation_cv_train(nfold, cv_id, obs_points,
    //     obs_unitvec,
    //                                     obs_sigma, dvec, nsar, ngnss);

    //     std::vector<std::vector<double>> obs_points_val;
    //     std::vector<std::vector<double>> obs_unitvec_val;
    //     std::vector<double> obs_sigma_val;
    //     std::vector<double> dvec_val;
    //     int nsar_val, ngnss_val;
    //     init::read_observation_cv_valid(nfold, cv_id, obs_points_val,
    //                                     obs_unitvec_val, obs_sigma_val,
    //                                     dvec_val, nsar_val, ngnss_val);

    //     std::vector<std::vector<double>> range = {
    //         {-10, 10}, {-30, 0}, {-30, -1}, {-20, 20},
    //         {50, 90},  {-2, 2},  {-2, 2},   {-10, -2}};
    //     std::string output_dir_cv =
    //         output_dir + "cv" + std::to_string(cv_id) + "/";
    //     std::string op = "mkdir -p " + output_dir_cv;
    //     system(op.c_str());

    //     std::vector<std::vector<double>> particles(nparticle_fault);
    //     smc_fault::smc_exec(particles, output_dir_cv, range, nparticle_fault,
    //                         cny_fault, coor_fault, obs_points, dvec,
    //                         obs_unitvec, obs_sigma, leta, node_to_elem,
    //                         id_dof, lmat_index, lmat_val, llmat, nsar,
    //                         ngnss);
    //     double cv_score = 0.;
    //     std::vector<double> cv_score_fault(nparticle_fault);
    // #pragma omp parallel for schedule(dynamic)
    //     for (int iparticle = 0; iparticle < nparticle_fault; iparticle++) {
    //         std::cout << "iparticle: " << iparticle << std::endl;
    //         auto particle = particles.at(iparticle);
    //         double xf = particle.at(0);
    //         double yf = particle.at(1);
    //         double zf = particle.at(2);
    //         double strike = particle.at(3);
    //         double dip = particle.at(4);
    //         double log_sigma_sar2 = particle.at(5);
    //         double log_sigma_gnss2 = particle.at(6);
    //         double log_alpha2 = particle.at(7);
    //         // Calculate greens function for the sampled fault
    //         auto gmat = gfunc::calc_greens_func(
    //             cny_fault, coor_fault, obs_points, obs_unitvec, leta, xf, yf,
    //             zf, strike, dip, node_to_elem, id_dof, nsar, ngnss);

    //         std::vector<double> gmat_flat(gmat.size() * gmat.at(0).size());
    //         for (int i = 0; i < gmat.size(); i++) {
    //             for (int j = 0; j < gmat.at(0).size(); j++) {
    //                 gmat_flat.at(i * gmat.at(0).size() + j) =
    //                 gmat.at(i).at(j);
    //             }
    //         }

    //         // diag component of Sigma
    //         //  (variance matrix for the likelihood function of slip)
    //         std::vector<double> sigma2_full(obs_sigma.size());
    //         for (int i = 0; i < nsar; i++) {
    //             sigma2_full.at(i) =
    //                 pow(obs_sigma.at(i), 2.) * exp(log_sigma_sar2);
    //         }
    //         for (int i = 0; i < ngnss; i++) {
    //             for (int j = 0; j < 3; j++) {
    //                 sigma2_full.at(nsar + 3 * i + j) =
    //                     pow(obs_sigma.at(nsar + 3 * i + j), 2.) *
    //                     exp(log_sigma_gnss2);
    //             }
    //         }

    //         // Calculate greens function for the sampled fault
    //         auto gmat_val = gfunc::calc_greens_func(
    //             cny_fault, coor_fault, obs_points_val, obs_unitvec_val, leta,
    //             xf, yf, zf, strike, dip, node_to_elem, id_dof, nsar_val,
    //             ngnss_val);

    //         std::vector<double> gmat_flat_val(gmat_val.size() *
    //                                           gmat_val.at(0).size());
    //         for (int i = 0; i < gmat_val.size(); i++) {
    //             for (int j = 0; j < gmat_val.at(0).size(); j++) {
    //                 gmat_flat_val.at(i * gmat_val.at(0).size() + j) =
    //                     gmat_val.at(i).at(j);
    //             }
    //         }

    //         // diag component of Sigma
    //         //  (variance matrix for the likelihood function of slip)
    //         std::vector<double> sigma2_full_val(obs_sigma_val.size());
    //         for (int i = 0; i < nsar_val; i++) {
    //             sigma2_full_val.at(i) =
    //                 pow(obs_sigma_val.at(i), 2.) * exp(log_sigma_sar2);
    //         }
    //         for (int i = 0; i < ngnss_val; i++) {
    //             for (int j = 0; j < 3; j++) {
    //                 sigma2_full_val.at(nsar_val + 3 * i + j) =
    //                     pow(obs_sigma_val.at(nsar_val + 3 * i + j), 2.) *
    //                     exp(log_sigma_gnss2);
    //             }
    //         }

    //         // Sequential Monte Carlo sampling for slip
    //         // calculate negative log of likelihood
    //         std::vector<std::vector<double>> particles_slip(nparticle_slip);
    //         smc_slip::smc_exec(particles_slip, output_dir_cv, nparticle_slip,
    //                            dvec, obs_sigma, sigma2_full, gmat,
    //                            log_sigma_sar2, log_sigma_gnss2, nsar, ngnss,
    //                            log_alpha2, lmat_index, lmat_val, llmat,
    //                            id_dof);

    //         for (int islip = 0; islip < nparticle_slip; islip++) {
    //             auto svec = particles_slip.at(islip);
    //             double delta_norm = 0.;
    //             smc_slip::calc_likelihood(svec, dvec_val, obs_sigma_val,
    //                                       sigma2_full_val, gmat_flat_val,
    //                                       log_sigma_sar2, log_sigma_gnss2,
    //                                       nsar_val, ngnss_val, delta_norm);
    //             cv_score_fault.at(iparticle) += delta_norm;
    //         }
    //         cv_score_fault.at(iparticle) /= nparticle_slip;
    //     }
    //     for (int iparticle = 0; iparticle < nparticle_fault; iparticle++) {
    //         cv_score += cv_score_fault.at(iparticle);
    //     }
    //     cv_score /= nparticle_fault;
    //     ofs << cv_score << std::endl;
    // }

    // std::vector<std::vector<double>> particles;
    // std::ifstream ifs(output_dir + "22.csv");
    // for (int iter = 0; iter < nparticle_fault; iter++) {
    //     double xf, yf, zf, strike, dip, log_sigma_sar2, log_sigma_gnss2,
    //         log_alpha2, likelihood;
    //     ifs >> xf >> yf >> zf >> strike >> dip >> log_sigma_sar2 >>
    //         log_sigma_gnss2 >> log_alpha2 >> likelihood;
    //     auto particle = {
    //         xf,        yf, zf, strike, dip, log_sigma_sar2, log_sigma_gnss2,
    //         log_alpha2};
    //     particles.push_back(particle);
    // }
    // #pragma omp parallel for schedule(dynamic)
    // for (int iparticle = 0; iparticle < nparticle_fault; iparticle++) {
    //     std::cout << "iparticle: " << iparticle << std::endl;
    //     auto particle = particles.at(iparticle);
    //     double xf = particle.at(0);
    //     double yf = particle.at(1);
    //     double zf = particle.at(2);
    //     double strike = particle.at(3);
    //     double dip = particle.at(4);
    //     double log_sigma_sar2 = particle.at(5);
    //     double log_sigma_gnss2 = particle.at(6);
    //     double log_alpha2 = particle.at(7);
    //     // Calculate greens function for the sampled fault
    //     auto gmat = gfunc::calc_greens_func(
    //         cny_fault, coor_fault, obs_points, obs_unitvec, leta, xf, yf, zf,
    //         strike, dip, node_to_elem, id_dof, nsar, ngnss);

    //     // diag component of Sigma
    //     //  (variance matrix for the likelihood function of slip)
    //     std::vector<double> sigma2_full(obs_sigma.size());
    //     for (int i = 0; i < nsar; i++) {
    //         sigma2_full.at(i) = pow(obs_sigma.at(i), 2.) *
    //         exp(log_sigma_sar2);
    //     }
    //     for (int i = 0; i < ngnss; i++) {
    //         for (int j = 0; j < 3; j++) {
    //             sigma2_full.at(nsar + 3 * i + j) =
    //                 pow(obs_sigma.at(nsar + 3 * i + j), 2.) *
    //                 exp(log_sigma_gnss2);
    //         }
    //     }

    //     // Sequential Monte Carlo sampling for slip
    //     // calculate negative log of likelihood
    //     std::vector<std::vector<double>> particles_slip(nparticle_slip);
    //     std::string dir =
    //         output_dir + "output_slip" + std::to_string(iparticle) + "/";
    //     std::string op = "mkdir -p " + dir;
    //     std::cout << op << std::endl;
    //     system(op.c_str());
    //     double neglog = smc_slip::smc_exec(
    //         particles_slip, dir, nparticle_slip, dvec, obs_sigma,
    //         sigma2_full, gmat, log_sigma_sar2, log_sigma_gnss2, nsar, ngnss,
    //         log_alpha2, lmat_index, lmat_val, llmat, id_dof);
    // }
    // std::exit(0);

    // Execute SMC for slip while fixing the fault
    // std::vector<double> particle = {
    //     1.09527761e+00, -1.23328200e+01, -1.36120457e+01, 1.40388275e+00,
    //     7.46067094e+01, 1.54248111e-01,  -1.19578646e-01, -2.29941335e+00};
    // double likelihood = smc_fault::calc_likelihood(
    //     particle, cny_fault, coor_fault, dvec, obs_points, obs_unitvec,
    //     obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss, lmat_index,
    //     lmat_val, llmat, nparticle_slip);
    // std::cout << " result: " << likelihood << std::endl;
    // std::exit(0);

    // range for xf, yf, zf, strike, dip, log_sigma2_sar, log_sigma2_gnss,
    // log_alpha2
    std::vector<std::vector<double>> range = {{-10, 10}, {-30, 0}, {-30, -1},
                                              {-20, 20}, {50, 90}, {-2, 2},
                                              {-2, 2},   {-10, -2}};

    // sequential monte carlo sampling for fault parameters
    // numbers of samples for approximation of distributions
    std::vector<std::vector<double>> particles;
    smc_fault::smc_exec(particles, output_dir, range, nparticle_fault,
                        cny_fault, coor_fault, obs_points, dvec, obs_unitvec,
                        obs_sigma, leta, node_to_elem, id_dof, lmat_index,
                        lmat_val, llmat, nsar, ngnss, nparticle_slip, myid,
                        numprocs);
    MPI_Finalize();
}
