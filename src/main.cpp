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

void aggregate_slip(
    const double &block_size, const std::string &fault_filepath,
    const std::string output_dir, const int &nparticle_fault,
    const int &nparticle_slip, const int &ndim, const double &lxi,
    const double &leta, const std::vector<double> &dvec,
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<std::vector<double>> &obs_unitvec,
    const std::vector<double> &obs_sigma,
    const std::unordered_map<int, std::vector<int>> &node_to_elem,
    const std::vector<int> &id_dof, const std::vector<int> lmat_index,
    const std::vector<double> lmat_val,
    const std::vector<std::vector<double>> llmat, const int &nsar,
    const int &ngnss, const int &myid, const int &numprocs);

void cross_validation(
    const std::string &output_dir, const int &nfold, const int &nparticle_fault,
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault, const double &leta,
    const std::unordered_map<int, std::vector<int>> &node_to_elem,
    const std::vector<int> &id_dof, const std::vector<int> &lmat_index,
    const std::vector<double> &lmat_val, const int &ndim,
    const std::vector<std::vector<double>> &llmat, const int &nparticle_slip,
    const int &myid, const int &numprocs);

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
    int nxi = lxi / 2.5;
    int neta = leta / 2.5;

    // generate output dir
    std::string output_dir =
        "output_cv/output_" + lxi_str + "_" + leta_str + "/cv/";
    // std::string output_dir = "output_cvtest/";
    std::string op = "mkdir -p " + output_dir;
    system(op.c_str());
    const int nparticle_slip = 10000;
    const int nparticle_fault = 2000;
    // set fault geometry
    // cny_fault[patch_id] = {node_id}
    std::vector<std::vector<int>> cny_fault;
    // coor_fault[node_id] = {node_coordinate}
    std::vector<std::vector<double>> coor_fault;
    // node_to_elem[node_id] = {patch_id containing the node}
    std::unordered_map<int, std::vector<int>> node_to_elem;
    // id_dof = {node_id which have degree of freedom}
    // slip value at node on the edge is fixed to be zero, no degree of
    // freedom.
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

    const int ndim = 8;
    const int nfold = 4;
    cross_validation(output_dir, nfold, nparticle_fault, cny_fault, coor_fault,
                     leta, node_to_elem, id_dof, lmat_index, lmat_val, ndim,
                     llmat, nparticle_slip, myid, numprocs);

    // // Execute SMC for slip while fixing the fault
    // std::vector<double> particle = {-1.29926, -23.2216,  -18.8675, 5.53965,
    //                                 57.6866,  -0.934742, 1.17758,  -6.21048};
    // double st_time, en_time;
    // st_time = MPI_Wtime();
    // double likelihood = smc_fault::calc_likelihood(
    //     particle, cny_fault, coor_fault, dvec, obs_points, obs_unitvec,
    //     obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss, lmat_index,
    //     lmat_val, llmat, nparticle_slip);
    // en_time = MPI_Wtime();
    // std::cout << " result: " << likelihood << std::endl;
    // std::cout << " etime: " << en_time - st_time << std::endl;
    // std::exit(0);
    //
    // std::vector<std::vector<double>> particles;
    // std::ifstream ifs(output_dir + "22.csv");
    // for (int iter = 0; iter < nparticle_fault; iter++) {
    //     double xf, yf, zf, strike, dip, log_sigma_sar2, log_sigma_gnss2,
    //         log_alpha2, likelihood;
    //     ifs >> xf >> yf >> zf >> strike >> dip >> log_sigma_sar2 >>
    //         log_sigma_gnss2 >> log_alpha2 >> likelihood;
    //     auto particle = {
    //         xf,        yf, zf, strike, dip, log_sigma_sar2,
    //         log_sigma_gnss2, log_alpha2};
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
    //         cny_fault, coor_fault, obs_points, obs_unitvec, leta, xf, yf,
    //         zf, strike, dip, node_to_elem, id_dof, nsar, ngnss);

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
    //         sigma2_full, gmat, log_sigma_sar2, log_sigma_gnss2, nsar,
    //         ngnss, log_alpha2, lmat_index, lmat_val, llmat, id_dof);
    // }
    // std::exit(0);

    // range for xf, yf, zf, strike, dip, log_sigma2_sar, log_sigma2_gnss,
    // log_alpha2
    // std::vector<std::vector<double>> range = {{-10, 10}, {-30, 0}, {-30, -1},
    //                                           {-20, 20}, {50, 90}, {-2, 2},
    //                                           {-2, 2},   {-10, -2}};

    // sequential monte carlo sampling for fault parameters
    // numbers of samples for approximation of distributions
    // std::vector<std::vector<double>> particles;
    // smc_fault::smc_exec(particles, output_dir, range, nparticle_fault,
    //                     cny_fault, coor_fault, obs_points, dvec,
    //                     obs_unitvec, obs_sigma, leta, node_to_elem,
    //                     id_dof, lmat_index, lmat_val, llmat, nsar, ngnss,
    //                     nparticle_slip, myid, numprocs);
    // aggregate_slip(2., output_dir + "21.csv", output_dir, nparticle_fault,
    //                nparticle_slip, range.size(), lxi, leta, dvec, cny_fault,
    //                coor_fault, obs_points, obs_unitvec, obs_sigma,
    //                node_to_elem, id_dof, lmat_index, lmat_val, llmat, nsar,
    //                ngnss, myid, numprocs);
    MPI_Finalize();
}

void aggregate_slip(
    const double &block_size, const std::string &fault_filepath,
    const std::string output_dir, const int &nparticle_fault,
    const int &nparticle_slip, const int &ndim, const double &lxi,
    const double &leta, const std::vector<double> &dvec,
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<std::vector<double>> &obs_unitvec,
    const std::vector<double> &obs_sigma,
    const std::unordered_map<int, std::vector<int>> &node_to_elem,
    const std::vector<int> &id_dof, const std::vector<int> lmat_index,
    const std::vector<double> lmat_val,
    const std::vector<std::vector<double>> llmat, const int &nsar,
    const int &ngnss, const int &myid, const int &numprocs) {
    const int &work_size = nparticle_fault / numprocs;
    std::string aggregate_dir = output_dir + "aggregate/";
    if (myid == 0) {
        std::string op = "mkdir -p " + aggregate_dir;
        std::cout << op << std::endl;
        system(op.c_str());
        op = "rm " + aggregate_dir + "*.dat";
        std::cout << op << std::endl;
        system(op.c_str());
    }

    std::ifstream ifs(fault_filepath);
    std::vector<double> particles_flat(work_size * ndim);
    std::vector<double> particles_flat_all;
    double xmax, xmin, ymax, ymin, zmax, zmin;
    if (myid == 0) {
        particles_flat_all.resize(nparticle_fault * ndim);
        double dum;
        for (int iparticle = 0; iparticle < nparticle_fault; iparticle++) {
            ifs >> particles_flat_all[iparticle * ndim + 0] >>
                particles_flat_all[iparticle * ndim + 1] >>
                particles_flat_all[iparticle * ndim + 2] >>
                particles_flat_all[iparticle * ndim + 3] >>
                particles_flat_all[iparticle * ndim + 4] >>
                particles_flat_all[iparticle * ndim + 5] >>
                particles_flat_all[iparticle * ndim + 6] >>
                particles_flat_all[iparticle * ndim + 7] >> dum;
        }
        // xmax = particles_flat_all[0];
        // xmin = particles_flat_all[0];
        // ymax = particles_flat_all[1];
        // ymin = particles_flat_all[1];
        // zmax = particles_flat_all[2];
        // zmin = particles_flat_all[2];
        // for (int iparticle = 0; iparticle < nparticle_fault; iparticle++)
        // {
        //     double xf = particles_flat_all[iparticle * ndim + 0];
        //     double yf = particles_flat_all[iparticle * ndim + 1];
        //     double zf = particles_flat_all[iparticle * ndim + 2];
        //     double strike = particles_flat_all[iparticle * ndim + 3];
        //     double dip = particles_flat_all[iparticle * ndim + 4];
        //     const double dip_rad = dip / 180. * M_PI;
        //     const double strike_rad = strike / 180. * M_PI;
        //     for (double xi : {0., lxi}) {
        //         for (double eta : {0., leta}) {
        //             double x = xf +
        //                        (leta - eta) * cos(dip_rad) *
        //                        cos(strike_rad)
        //                        + xi * sin(strike_rad);
        //             double y = yf -
        //                        (leta - eta) * cos(dip_rad) *
        //                        sin(strike_rad)
        //                        + xi * cos(strike_rad);
        //             double z = zf - (leta - eta) * sin(dip_rad);
        //             xmax = fmax(x, xmax);
        //             xmin = fmin(x, xmin);
        //             ymax = fmax(y, ymax);
        //             ymin = fmin(y, ymin);
        //             zmax = fmax(z, zmax);
        //             zmin = fmin(z, zmin);
        //         }
        //     }
        // }
        xmin = -2.;
        xmax = 12;
        ymin = -20.;
        ymax = 10;
        zmin = -40.;
        zmax = -8;
    }
    MPI_Scatter(&particles_flat_all[0], work_size * ndim, MPI_DOUBLE,
                &particles_flat[0], work_size * ndim, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Bcast(&xmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xmin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ymax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ymin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zmin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int nx = (xmax - xmin) / block_size + 1;
    int ny = (ymax - ymin) / block_size + 1;
    int nz = (zmax - zmin) / block_size + 1;

    if (myid == 0) {
        std::ofstream ofs(aggregate_dir + "block_coord.dat");
        for (int ix = 0; ix < nx; ix++) {
            for (int iy = 0; iy < ny; iy++) {
                for (int iz = 0; iz < nz; iz++) {
                    ofs << ix << "_" << iy << "_" << iz << " "
                        << xmin + block_size * ix << " "
                        << ymin + block_size * iy << " "
                        << zmin + block_size * iz << std::endl;
                }
            }
        }
    }

    std::vector<std::vector<double>> block_res(nx * ny * nz);
    const int nnode = coor_fault.size();
    // for (int iparticle_fault = 0; iparticle_fault < 1;
    // iparticle_fault++) {
    double st_time, en_time;
    st_time = MPI_Wtime();
    for (int iparticle_fault = 0; iparticle_fault < work_size;
         iparticle_fault++) {
        std::vector<double> particle(ndim);
        for (int idim = 0; idim < ndim; idim++) {
            particle.at(idim) =
                particles_flat.at(iparticle_fault * ndim + idim);
        }
        double xf = particle.at(0);
        double yf = particle.at(1);
        double zf = particle.at(2);
        double strike = particle.at(3);
        double dip = particle.at(4);
        double log_sigma_sar2 = particle.at(5);
        double log_sigma_gnss2 = particle.at(6);
        double log_alpha2 = particle.at(7);
        const double dip_rad = dip / 180. * M_PI;
        const double strike_rad = strike / 180. * M_PI;
        // Calculate greens function for the sampled fault
        auto gmat = gfunc::calc_greens_func(
            cny_fault, coor_fault, obs_points, obs_unitvec, leta, xf, yf, zf,
            strike, dip, node_to_elem, id_dof, nsar, ngnss);

        // diag component of Sigma
        //  (variance matrix for the likelihood function of slip)
        std::vector<double> sigma2_full(obs_sigma.size());
        for (int i = 0; i < nsar; i++) {
            sigma2_full.at(i) = pow(obs_sigma.at(i), 2.) * exp(log_sigma_sar2);
        }
        for (int i = 0; i < ngnss; i++) {
            for (int j = 0; j < 3; j++) {
                sigma2_full.at(nsar + 3 * i + j) =
                    pow(obs_sigma.at(nsar + 3 * i + j), 2.) *
                    exp(log_sigma_gnss2);
            }
        }

        // Sequential Monte Carlo sampling for slip
        // calculate negative log of likelihood
        std::vector<std::vector<double>> particles_slip(nparticle_slip);
        std::string dir = output_dir + "output_slip" +
                          std::to_string(work_size * myid + iparticle_fault) +
                          "/";
        std::string op = "mkdir -p " + dir;
        // std::cout << op << std::endl;
        system(op.c_str());
        double neglog = smc_slip::smc_exec(
            particles_slip, dir, nparticle_slip, dvec, obs_sigma, sigma2_full,
            gmat, log_sigma_sar2, log_sigma_gnss2, nsar, ngnss, log_alpha2,
            lmat_index, lmat_val, llmat, id_dof);
        // std::cout << "iparticle_fault = " << work_size * myid +
        // iparticle_fault
        //           << " neglog: " << neglog << std::endl;

        // for (int iparticle_slip = 0; iparticle_slip <
        // nparticle_slip;
        //      iparticle_slip++) {
        std::vector<std::vector<double>> slip(nparticle_slip,
                                              std::vector<double>(2 * nnode));
        for (int iparticle_slip = 0; iparticle_slip < nparticle_slip;
             iparticle_slip++) {
            const std::vector<double> particle_slip =
                particles_slip.at(iparticle_slip);
            for (int idof = 0; idof < id_dof.size(); idof++) {
                int inode = id_dof.at(idof);
                slip.at(iparticle_slip).at(2 * inode + 0) =
                    particle_slip.at(2 * idof + 0);
                slip.at(iparticle_slip).at(2 * inode + 1) =
                    particle_slip.at(2 * idof + 1);
            }
        }

        for (int ie = 0; ie < cny_fault.size(); ie++) {
            auto node_id = cny_fault[ie];
            std::vector<double> xinode(4);
            std::vector<double> etanode(4);
            std::vector<std::vector<double>> sxinode(nparticle_slip,
                                                     std::vector<double>(4));
            std::vector<std::vector<double>> setanode(nparticle_slip,
                                                      std::vector<double>(4));
            for (int inode = 0; inode < node_id.size(); inode++) {
                xinode.at(inode) = coor_fault.at(node_id.at(inode)).at(0);
                etanode.at(inode) = coor_fault.at(node_id.at(inode)).at(1);
                for (int iparticle_slip = 0; iparticle_slip < nparticle_slip;
                     iparticle_slip++) {
                    sxinode.at(iparticle_slip).at(inode) =
                        slip.at(iparticle_slip).at(2 * node_id.at(inode) + 0);
                    setanode.at(iparticle_slip).at(inode) =
                        slip.at(iparticle_slip).at(2 * node_id.at(inode) + 1);
                }
            }
            for (double r1 : {-0.667, 0., 0.667}) {
                for (double r2 : {-0.667, 0., 0.667}) {
                    std::vector<double> nvec = {
                        (1. - r1) * (1 - r2) / 4., (1. + r1) * (1 - r2) / 4.,
                        (1. + r1) * (1 + r2) / 4., (1. - r1) * (1 + r2) / 4.};
                    double xi = 0.;
                    double eta = 0.;
                    std::vector<double> sxi(nparticle_slip);
                    std::vector<double> seta(nparticle_slip);
                    std::vector<double> sn(nparticle_slip);
                    for (int inode = 0; inode < node_id.size(); inode++) {
                        xi += nvec[inode] * xinode[inode];
                        eta += nvec[inode] * etanode[inode];
                        for (int iparticle_slip = 0;
                             iparticle_slip < nparticle_slip;
                             iparticle_slip++) {
                            sxi[iparticle_slip] +=
                                nvec[inode] * sxinode[iparticle_slip][inode];
                            seta[iparticle_slip] +=
                                nvec[inode] * setanode[iparticle_slip][inode];
                        }
                    }
                    for (int iparticle_slip = 0;
                         iparticle_slip < nparticle_slip; iparticle_slip++) {
                        sn[iparticle_slip] = sqrt(pow(sxi[iparticle_slip], 2) +
                                                  pow(seta[iparticle_slip], 2));
                    }
                    double x = xf +
                               (leta - eta) * cos(dip_rad) * cos(strike_rad) +
                               xi * sin(strike_rad);
                    double y = yf -
                               (leta - eta) * cos(dip_rad) * sin(strike_rad) +
                               xi * cos(strike_rad);
                    double z = zf - (leta - eta) * sin(dip_rad);
                    int ix = (x - xmin) / block_size;
                    int iy = (y - ymin) / block_size;
                    int iz = (z - zmin) / block_size;
                    for (int iparticle_slip = 0;
                         iparticle_slip < nparticle_slip; iparticle_slip++) {
                        block_res.at(ix * ny * nz + iy * nz + iz)
                            .push_back(sn[iparticle_slip]);
                    }
                }
            }
        }
    }
    en_time = MPI_Wtime();
    std::cout << "Myid: " << myid << " calc_etime: " << en_time - st_time
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    st_time = MPI_Wtime();
    for (int iproc = 0; iproc < numprocs; iproc++) {
        for (int block_id = myid - iproc; block_id < nx * ny * nz;
             block_id += numprocs) {
            if (block_id < 0) {
                continue;
            }
            int block_id_t = block_id;
            int ix = block_id_t / (ny * nz);
            block_id_t -= ix * (ny * nz);
            int iy = block_id_t / nz;
            block_id_t -= iy * nz;
            int iz = block_id_t;
            // std::cout << "block_id: " << block_id << " myid: " << myid
            //           << std::endl;
            std::string filename = aggregate_dir + std::to_string(ix) + "_" +
                                   std::to_string(iy) + "_" +
                                   std::to_string(iz) + ".dat";
            int size = block_res.at(block_id).size();
            if (size > 0) {
                FILE *fp = fopen(filename.c_str(), "a");
                for (int is = 0; is < size; is++) {
                    fprintf(fp, "%f\n", block_res.at(block_id).at(is));
                }
                fclose(fp);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    en_time = MPI_Wtime();
    std::cout << "Myid: " << myid << " output_etime: " << en_time - st_time
              << std::endl;
}

void cross_validation(
    const std::string &output_dir, const int &nfold, const int &nparticle_fault,
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault, const double &leta,
    const std::unordered_map<int, std::vector<int>> &node_to_elem,
    const std::vector<int> &id_dof, const std::vector<int> &lmat_index,
    const std::vector<double> &lmat_val, const int &ndim,
    const std::vector<std::vector<double>> &llmat, const int &nparticle_slip,
    const int &myid, const int &numprocs) {
    const int work_size = nparticle_fault / numprocs;
    std::ofstream ofs(output_dir + "cv_score.dat");
    for (int cv_id = 0; cv_id < nfold; cv_id++) {
        std::vector<std::vector<double>> obs_points;
        std::vector<std::vector<double>> obs_unitvec;
        std::vector<double> obs_sigma;
        std::vector<double> dvec;
        int nsar, ngnss;
        init::read_observation_cv_train(nfold, cv_id, obs_points, obs_unitvec,
                                        obs_sigma, dvec, nsar, ngnss);

        std::vector<std::vector<double>> obs_points_val;
        std::vector<std::vector<double>> obs_unitvec_val;
        std::vector<double> obs_sigma_val;
        std::vector<double> dvec_val;
        int nsar_val, ngnss_val;
        init::read_observation_cv_valid(nfold, cv_id, obs_points_val,
                                        obs_unitvec_val, obs_sigma_val,
                                        dvec_val, nsar_val, ngnss_val);

        // std::vector<double> particles_flat_t;
        // std::vector<double> work_particles_flat_t(work_size * ndim);
        // if (myid == 0) {
        //     particles_flat_t.resize(nparticle_fault * ndim);
        //     std::ifstream ifs("tmp/init_samples.dat");
        //     for (int iparticle = 0; iparticle < nparticle_fault; iparticle++)
        //     {
        //         for (int idim = 0; idim < ndim; idim++) {
        //             ifs >> particles_flat_t.at(iparticle * ndim + idim);
        //         }
        //     }
        // }
        // MPI_Scatter(&particles_flat_t[0], work_size * ndim, MPI_DOUBLE,
        //             &work_particles_flat_t.at(0), work_size * ndim,
        //             MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Barrier(MPI_COMM_WORLD);
        // double st_time = MPI_Wtime();
        // #pragma omp parallel for
        // for (int iparticle = 0; iparticle < work_size; iparticle++) {
        //     std::vector<double> particle(ndim);
        //     for (int idim = 0; idim < ndim; idim++) {
        //         particle.at(idim) =
        //             work_particles_flat_t.at(iparticle * ndim + idim);
        //     }
        //     // calculate negative log likelihood for the sample
        //     double st_time, en_time;
        //     st_time = MPI_Wtime();
        //     double likelihood = smc_fault::calc_likelihood(
        //         particle, cny_fault, coor_fault, dvec, obs_points,
        //         obs_unitvec, obs_sigma, leta, node_to_elem, id_dof, nsar,
        //         ngnss, lmat_index, lmat_val, llmat, nparticle_slip);
        //     en_time = MPI_Wtime();
        //     // std::cout << "myid: " << myid << " etime: " << en_time -
        //     st_time
        //     //           << std::endl;
        //     printf("myid: %d, etime: %f\n", myid, en_time - st_time);
        //     // std::cout << "iparticle: " << iparticle + myid * work_size
        //     //           << " likelihood: " <<
        //     //           work_init_likelihood.at(iparticle)
        //     //           << std::endl;
        // }
        // MPI_Barrier(MPI_COMM_WORLD);
        // double en_time = MPI_Wtime();
        // if (myid == 0) {
        //     printf("total etime: %f\n", en_time - st_time);
        // }

        std::vector<std::vector<double>> range = {
            {-10, 10}, {-30, 0}, {-30, -1}, {-20, 20},
            {50, 90},  {-2, 2},  {-2, 2},   {-10, 2}};
        std::string output_dir_cv =
            output_dir + "cv" + std::to_string(cv_id) + "/";
        std::string op = "mkdir -p " + output_dir_cv;
        system(op.c_str());

        std::vector<double> particles_flat;
        smc_fault::smc_exec(particles_flat, output_dir_cv, range,
                            nparticle_fault, cny_fault, coor_fault, obs_points,
                            dvec, obs_unitvec, obs_sigma, leta, node_to_elem,
                            id_dof, lmat_index, lmat_val, llmat, nsar, ngnss,
                            nparticle_slip, myid, numprocs);
        // MPI_Barrier(MPI_COMM_WORLD);
        // MPI_Finalize();
        // exit(0);

        std::vector<double> work_particles_flat(work_size * ndim);
        MPI_Scatter(&particles_flat[0], work_size * ndim, MPI_DOUBLE,
                    &work_particles_flat.at(0), work_size * ndim, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);

        double cv_score;
        std::vector<double> cv_score_fault(work_size);
        for (int iparticle = 0; iparticle < work_size; iparticle++) {
            std::cout << "iparticle: " << iparticle + work_size * myid
                      << std::endl;
            std::vector<double> particle(ndim);
            for (int idim = 0; idim < ndim; idim++) {
                particle.at(idim) =
                    work_particles_flat.at(iparticle * ndim + idim);
            }
            double xf = particle.at(0);
            double yf = particle.at(1);
            double zf = particle.at(2);
            double strike = particle.at(3);
            double dip = particle.at(4);
            double log_sigma_sar2 = particle.at(5);
            double log_sigma_gnss2 = particle.at(6);
            double log_alpha2 = particle.at(7);
            // Calculate greens function for the sampled fault
            auto gmat = gfunc::calc_greens_func(
                cny_fault, coor_fault, obs_points, obs_unitvec, leta, xf, yf,
                zf, strike, dip, node_to_elem, id_dof, nsar, ngnss);

            std::vector<double> gmat_flat(gmat.size() * gmat.at(0).size());
            for (int i = 0; i < gmat.size(); i++) {
                for (int j = 0; j < gmat.at(0).size(); j++) {
                    gmat_flat.at(i * gmat.at(0).size() + j) = gmat.at(i).at(j);
                }
            }

            // diag component of Sigma
            //  (variance matrix for the likelihood function of slip)
            std::vector<double> sigma2_full(obs_sigma.size());
            for (int i = 0; i < nsar; i++) {
                sigma2_full.at(i) =
                    pow(obs_sigma.at(i), 2.) * exp(log_sigma_sar2);
            }
            for (int i = 0; i < ngnss; i++) {
                for (int j = 0; j < 3; j++) {
                    sigma2_full.at(nsar + 3 * i + j) =
                        pow(obs_sigma.at(nsar + 3 * i + j), 2.) *
                        exp(log_sigma_gnss2);
                }
            }

            // Calculate greens function for the sampled fault
            auto gmat_val = gfunc::calc_greens_func(
                cny_fault, coor_fault, obs_points_val, obs_unitvec_val, leta,
                xf, yf, zf, strike, dip, node_to_elem, id_dof, nsar_val,
                ngnss_val);

            std::vector<double> gmat_flat_val(gmat_val.size() *
                                              gmat_val.at(0).size());
            for (int i = 0; i < gmat_val.size(); i++) {
                for (int j = 0; j < gmat_val.at(0).size(); j++) {
                    gmat_flat_val.at(i * gmat_val.at(0).size() + j) =
                        gmat_val.at(i).at(j);
                }
            }

            // diag component of Sigma
            //  (variance matrix for the likelihood function of slip)
            std::vector<double> sigma2_full_val(obs_sigma_val.size());
            for (int i = 0; i < nsar_val; i++) {
                sigma2_full_val.at(i) =
                    pow(obs_sigma_val.at(i), 2.) * exp(log_sigma_sar2);
            }
            for (int i = 0; i < ngnss_val; i++) {
                for (int j = 0; j < 3; j++) {
                    sigma2_full_val.at(nsar_val + 3 * i + j) =
                        pow(obs_sigma_val.at(nsar_val + 3 * i + j), 2.) *
                        exp(log_sigma_gnss2);
                }
            }

            // Sequential Monte Carlo sampling for slip
            // calculate negative log of likelihood
            std::vector<std::vector<double>> particles_slip(nparticle_slip);
            smc_slip::smc_exec(particles_slip, output_dir_cv, nparticle_slip,
                               dvec, obs_sigma, sigma2_full, gmat,
                               log_sigma_sar2, log_sigma_gnss2, nsar, ngnss,
                               log_alpha2, lmat_index, lmat_val, llmat, id_dof);

            for (int islip = 0; islip < nparticle_slip; islip++) {
                auto svec = particles_slip.at(islip);
                double delta_norm = 0.;
                smc_slip::calc_likelihood(svec, dvec_val, obs_sigma_val,
                                          sigma2_full_val, gmat_flat_val,
                                          log_sigma_sar2, log_sigma_gnss2,
                                          nsar_val, ngnss_val, delta_norm);
                cv_score_fault.at(iparticle) += delta_norm;
            }
            cv_score_fault.at(iparticle) /= nparticle_slip;
        }
        double cv_score_t = 0.;
        for (int iparticle = 0; iparticle < work_size; iparticle++) {
            cv_score_t += cv_score_fault.at(iparticle);
        }
        cv_score_t /= nparticle_fault;
        MPI_Reduce(&cv_score_t, &cv_score, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        if (myid == 0) {
            ofs << cv_score << std::endl;
        }
    }
}
