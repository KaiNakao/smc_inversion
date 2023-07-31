#include "smc_fault.hpp"

#include <cstdlib>
#include <fstream>

namespace smc_fault {

double calc_likelihood(const std::vector<double> &particle,
                       const std::vector<double> &dvec,
                       const std::vector<std::vector<double>> &obs_points,
                       const std::vector<std::vector<double>> &obs_unitvec,
                       const std::vector<double> &obs_sigma, const int &nsar,
                       const int &ngnss, const int nparticle_slip,
                       const double &max_slip, const int &nxi,
                       const int &neta) {
    double xf = particle.at(0);
    double yf = particle.at(1);
    double zf = particle.at(2);
    double strike = particle.at(3);
    double dip = particle.at(4);
    double log_sigma_sar2 = particle.at(5);
    double log_sigma_gnss2 = particle.at(6);
    double log_alpha2 = particle.at(7);
    double lxi = particle.at(8);
    double leta = particle.at(9);

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

    // double st_time, en_time;
    // st_time = MPI_Wtime();
    // Calculate greens function for the sampled fault
    auto gmat = gfunc::calc_greens_func(cny_fault, coor_fault, obs_points,
                                        obs_unitvec, leta, xf, yf, zf, strike,
                                        dip, node_to_elem, id_dof, nsar, ngnss);
    // en_time = MPI_Wtime();

    // diag component of Sigma
    //  (variance matrix for the likelihood function of slip)
    std::vector<double> sigma2_full(obs_sigma.size());
    for (int i = 0; i < nsar; i++) {
        sigma2_full.at(i) = pow(obs_sigma.at(i), 2.) * exp(log_sigma_sar2);
    }
    for (int i = 0; i < ngnss; i++) {
        for (int j = 0; j < 3; j++) {
            sigma2_full.at(nsar + 3 * i + j) =
                pow(obs_sigma.at(nsar + 3 * i + j), 2.) * exp(log_sigma_gnss2);
        }
    }

    // Sequential Monte Carlo sampling for slip
    // calculate negative log of likelihood
    // st_time = MPI_Wtime();
    std::vector<std::vector<double>> particles_slip(nparticle_slip);
    double neglog = smc_slip::smc_exec(
        particles_slip, "output_slip/", nparticle_slip, dvec, obs_sigma,
        sigma2_full, gmat, log_sigma_sar2, log_sigma_gnss2, nsar, ngnss,
        log_alpha2, lmat_index, lmat_val, llmat, id_dof, max_slip);
    // en_time = MPI_Wtime();
    // std::cout << xf << " " << yf << " " << zf << " " << strike << " " << dip
    //           << " " << log_sigma_sar2 << " " << log_sigma_gnss2 << " "
    //           << log_alpha2 << " " << neglog << " " << en_time - st_time
    //           << std::endl;
    // std::cout << "smc etime: " << en_time - st_time << std::endl;
    // double maxslip =
    //     *std::max_element(particles_slip[0].begin(),
    //     particles_slip[0].end());
    // double sumslip = 0.;
    // for (int i = 0; i < particles_slip.at(0).size(); i++) {
    //     sumslip += particles_slip.at(0).at(i);
    // }
    // printf("max slip: %f\n", maxslip);
    // printf("sum slip: %f\n", sumslip);
    return neglog;
}

void sample_init_particles(std::vector<double> &particles_flat,
                           const int &nparticle, const int &ndim,
                           const std::vector<std::vector<double>> &range) {
    particles_flat.resize(nparticle * ndim);

    // probability distribution instance for generating samples from piror
    // (uniform distribution)
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::vector<std::uniform_real_distribution<>> dist_vec(range.size());
    for (int idim = 0; idim < range.size(); idim++) {
        std::uniform_real_distribution<> dist(range.at(idim).at(0),
                                              range.at(idim).at(1));
        dist_vec.at(idim) = dist;
    }

    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        for (int idim = 0; idim < ndim; idim++) {
            double x = dist_vec.at(idim)(engine);
            particles_flat.at(iparticle * ndim + idim) = x;
        }
    }

    return;
}

void work_eval_init_particles(
    const int &work_size, const int &ndim,
    const std::vector<double> &work_particles_flat,
    std::vector<double> &work_init_likelihood,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<double> &dvec,
    const std::vector<std::vector<double>> &obs_unitvec,
    const std::vector<double> &obs_sigma, const int &nsar, const int &ngnss,
    const int &nparticle_slip, const double &max_slip, const int &nxi,
    const int &neta, const int &myid) {
#pragma omp parallel for
    for (int iparticle = 0; iparticle < work_size; iparticle++) {
        std::vector<double> particle(ndim);
        for (int idim = 0; idim < ndim; idim++) {
            particle.at(idim) = work_particles_flat.at(iparticle * ndim + idim);
        }
        // calculate negative log likelihood for the sample
        work_init_likelihood.at(iparticle) =
            calc_likelihood(particle, dvec, obs_points, obs_unitvec, obs_sigma,
                            nsar, ngnss, nparticle_slip, max_slip, nxi, neta);
        // std::cout << "iparticle: " << iparticle + myid * work_size
        //           << " likelihood: " <<
        //           work_init_likelihood.at(iparticle)
        //           << std::endl;
    }
}  // namespace smc_fault

std::vector<double> calc_mean_std_vector(const std::vector<double> &vec) {
    // ret = {mean, std} over the component of vector
    std::vector<double> ret;
    double mean = 0.;
    for (int i = 0; i < vec.size(); i++) {
        mean += vec.at(i);
    }
    mean /= vec.size();
    double var = 0.;
    for (int i = 0; i < vec.size(); i++) {
        var += pow(vec.at(i) - mean, 2);
    }
    var /= vec.size();
    double std = sqrt(var);
    ret = {mean, std};

    return ret;
}

double find_next_gamma(const double &gamma_prev,
                       std::vector<double> &likelihood_ls,
                       std::vector<double> &weights) {
    int nparticle = weights.size();

    // find minimum of negative log likelihood
    double min_likelihood = likelihood_ls.at(0);
    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        min_likelihood = std::min(min_likelihood, likelihood_ls.at(iparticle));
    }

    double cv_threshold = 0.5;
    // binary search for the next gamma
    // such that c.o.v of the weight is equivalent to cv_threashold
    double lower = gamma_prev;
    double upper = 1.;
    double err = 1.;
    double gamma = 1.;
    while (err > pow(10, -8)) {
        gamma = (lower + upper) / 2.;

        double diff_gamma = gamma - gamma_prev;
        for (int iparticle = 0; iparticle < nparticle; iparticle++) {
            double likelihood = likelihood_ls.at(iparticle);
            // extract min_likelihood to avoid underflow of the weight
            weights.at(iparticle) =
                exp(-diff_gamma * (likelihood - min_likelihood));
        }

        // calculate c.o.v = mean / std
        auto mean_std = calc_mean_std_vector(weights);
        double cv = mean_std.at(1) / mean_std.at(0);

        if (cv > cv_threshold) {
            upper = gamma;
        } else {
            lower = gamma;
        }
        err = fabs(cv - cv_threshold);
        if (fabs(gamma - 1) < pow(10, -8)) {
            break;
        }
    }
    return gamma;
}

std::vector<double> normalize_weights(const std::vector<double> &weights) {
    int n_particle = weights.size();
    std::vector<double> ret(n_particle);
    double sum = 0;
    for (int n = 0; n < n_particle; n++) {
        sum += weights.at(n);
    }
    for (int n = 0; n < n_particle; n++) {
        ret.at(n) = weights.at(n) / sum;
    }
    return ret;
}

std::vector<double> calc_mean_particles(
    const std::vector<double> &particles_flat,
    const std::vector<double> &weights, const int &nparticle, const int &ndim) {
    std::vector<double> mean(ndim);
    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        for (int idim = 0; idim < ndim; idim++) {
            mean.at(idim) += weights.at(iparticle) *
                             particles_flat.at(iparticle * ndim + idim);
        }
    }
    std::cout << "mean:" << std::endl;
    for (int i = 0; i < ndim; i++) {
        std::cout << mean.at(i) << " ";
    }
    std::cout << std::endl;
    return mean;
}

std::vector<double> calc_cov_particles(
    const std::vector<double> &particles_flat,
    const std::vector<double> &weights, const std::vector<double> &mean,
    const int &nparticle, const int &ndim) {
    std::vector<double> cov_flat(ndim * ndim);
    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        for (int idim = 0; idim < ndim; idim++) {
            for (int jdim = 0; jdim < ndim; jdim++) {
                cov_flat.at(idim * ndim + jdim) +=
                    weights.at(iparticle) *
                    (particles_flat.at(iparticle * ndim + idim) -
                     mean.at(idim)) *
                    (particles_flat.at(iparticle * ndim + jdim) -
                     mean.at(jdim));
            }
        }
    }
    std::cout << "cov:" << std::endl;
    for (int i = 0; i < ndim; i++) {
        std::cout << cov_flat.at(i * ndim + i) << " ";
    }
    std::cout << std::endl;
    for (int idim = 0; idim < ndim; idim++) {
        for (int jdim = 0; jdim < ndim; jdim++) {
            cov_flat[idim * ndim + jdim] *= 0.04;
        }
    }

    // LAPACK function for LU decomposition of matrix
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', ndim, &cov_flat[0], ndim);
    return cov_flat;
}

std::vector<int> resample_particles(const int &nparticle,
                                    const std::vector<double> &weights) {
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());

    // resampling
    // list for the indice of original particle of each resampled particle
    double deno = nparticle;
    std::vector<double> uvec(nparticle);
    for (int n = 0; n < uvec.size(); n++) {
        std::uniform_real_distribution<> dist1(n / deno, (n + 1) / deno);
        uvec.at(n) = dist1(engine);
    }
    std::vector<double> cumsum(nparticle);
    cumsum.at(0) = weights.at(0);
    for (int n = 0; n < nparticle - 1; n++) {
        cumsum.at(n + 1) = cumsum.at(n) + weights.at(n + 1);
    }
    std::vector<int> assigned_num(nparticle, 0);
    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        auto it =
            std::lower_bound(cumsum.begin(), cumsum.end(), uvec.at(iparticle));
        int i = std::distance(cumsum.begin(), it);
        (assigned_num[i])++;
    }
    return assigned_num;
}

void reorder_to_send(std::vector<int> &assigned_num,
                     std::vector<double> &particles_flat, const int &nparticle,
                     const int &ndim, const int &numprocs,
                     const int &work_size) {
    std::vector<int> sorted_idx(nparticle);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
        return assigned_num.at(a) > assigned_num.at(b);
    });

    std::vector<double> tmp_particles_flat = particles_flat;
    std::vector<int> tmp_assigned_num = assigned_num;
    for (int iproc = 0; iproc < numprocs; iproc++) {
        for (int iparticle = 0; iparticle < work_size; iparticle++) {
            int id_org = sorted_idx.at(iparticle * numprocs + iproc);
            int id_new = iproc * work_size + iparticle;
            assigned_num.at(id_new) = tmp_assigned_num.at(id_org);
            for (int idim = 0; idim < ndim; idim++) {
                particles_flat.at(id_new * ndim + idim) =
                    tmp_particles_flat.at(id_org * ndim + idim);
            }
        }
    }
    return;
}

void work_mcmc_sampling(const std::vector<int> &work_assigned_num,
                        const std::vector<double> &work_particles_flat,
                        std::vector<double> &work_particles_flat_new,
                        std::vector<double> &work_likelihood_ls_new,
                        const int &work_size, const int &ndim,
                        const std::vector<double> &cov_flat,
                        const double &gamma,
                        const std::vector<std::vector<double>> &obs_points,
                        const std::vector<double> &dvec,
                        const std::vector<std::vector<double>> &obs_unitvec,
                        const std::vector<double> &obs_sigma, const int &nsar,
                        const int &ngnss, const int &nparticle_slip,
                        const double &max_slip, const int &nxi, const int &neta,
                        const int &myid) {
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    // probability distribution for MCCMC metropolis test
    std::uniform_real_distribution<> dist_metropolis(0., 1.);
    // standard normal distribution
    std::normal_distribution<> dist_stnorm(0., 1.);

    std::vector<int> id_start(work_size);
    for (int iparticle = 0; iparticle < work_size - 1; iparticle++) {
        id_start.at(iparticle + 1) =
            id_start.at(iparticle) + work_assigned_num.at(iparticle);
    }
    int sum_assigned =
        id_start.at(work_size - 1) + work_assigned_num.at(work_size - 1);
    work_particles_flat_new.resize(sum_assigned * ndim);
    work_likelihood_ls_new.resize(sum_assigned);
#pragma omp parallel for
    for (int iparticle = 0; iparticle < work_size; iparticle++) {
        int nassigned = work_assigned_num.at(iparticle);
        int istart = id_start.at(iparticle);
        std::vector<double> particle_cur(ndim);
        for (int idim = 0; idim < ndim; idim++) {
            particle_cur.at(idim) =
                work_particles_flat.at(iparticle * ndim + idim);
        }
        for (int jparticle = 0; jparticle < nassigned; jparticle++) {
            double likelihood_cur = calc_likelihood(
                particle_cur, dvec, obs_points, obs_unitvec, obs_sigma, nsar,
                ngnss, nparticle_slip, max_slip, nxi, neta);
            // propose particle_cand
            std::vector<double> rand(ndim);
            for (int idim = 0; idim < ndim; idim++) {
                rand.at(idim) = dist_stnorm(engine);
            }
            std::vector<double> particle_cand(ndim);
            for (int idim = 0; idim < ndim; idim++) {
                particle_cand[idim] = particle_cur[idim];
                for (int jdim = 0; jdim < idim + 1; jdim++) {
                    particle_cand[idim] +=
                        cov_flat[idim * ndim + jdim] * rand[jdim];
                }
            }

            // calculate negative log likelihood of the proposed
            // configuration
            double likelihood_cand = calc_likelihood(
                particle_cand, dvec, obs_points, obs_unitvec, obs_sigma, nsar,
                ngnss, nparticle_slip, max_slip, nxi, neta);
            double metropolis = dist_metropolis(engine);

            // metropolis test and check domain of definition
            if (particle_cand.at(4) < 90 && particle_cand.at(2) < 0 &&
                particle_cand.at(8) > 0 && particle_cand.at(9) > 0 &&
                exp(gamma * (likelihood_cur - likelihood_cand)) > metropolis) {
                // std::cout << "accepted likelihood: " << likelihood_cand
                //           << std::endl;
                particle_cur = particle_cand;
                likelihood_cur = likelihood_cand;
            } else {
                // std::cout << "rejected likelihood: " << likelihood_cand
                //           << std::endl;
            }

            // save to new particle list
            work_likelihood_ls_new.at(istart + jparticle) = likelihood_cur;
            for (int idim = 0; idim < ndim; idim++) {
                work_particles_flat_new.at((istart + jparticle) * ndim + idim) =
                    particle_cur.at(idim);
            }
        }
    }
    return;
}

void smc_exec(std::vector<double> &particles_flat,
              const std::string &output_dir,
              const std::vector<std::vector<double>> &range,
              const int &nparticle,
              const std::vector<std::vector<double>> &obs_points,
              const std::vector<double> &dvec,
              const std::vector<std::vector<double>> &obs_unitvec,
              const std::vector<double> &obs_sigma, const int &nsar,
              const int &ngnss, const int &nparticle_slip,
              const double &max_slip, const int &nxi, const int &neta,
              const int &myid, const int &numprocs) {
    MPI_Status status;
    const int ndim = range.size();
    const int work_size = nparticle / numprocs;
    // list for (negative log) likelihood for each particles
    std::vector<double> likelihood_ls;
    if (myid == 0) {
        likelihood_ls.resize(nparticle);
        // sampling from the prior distribution
        sample_init_particles(particles_flat, nparticle, ndim, range);
    }
    std::vector<double> work_particles_flat(work_size * ndim);
    std::vector<double> work_init_likelihood(work_size);
    MPI_Scatter(&particles_flat[0], work_size * ndim, MPI_DOUBLE,
                &work_particles_flat.at(0), work_size * ndim, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    work_eval_init_particles(work_size, ndim, work_particles_flat,
                             work_init_likelihood, obs_points, dvec,
                             obs_unitvec, obs_sigma, nsar, ngnss,
                             nparticle_slip, max_slip, nxi, neta, myid);
    MPI_Gather(&work_init_likelihood.at(0), work_size, MPI_DOUBLE,
               &likelihood_ls[0], work_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        // output result of stage 0
        std::ofstream ofs(output_dir + std::to_string(0) + ".csv");
        for (int iparticle = 0; iparticle < nparticle; iparticle++) {
            for (int idim = 0; idim < ndim; idim++) {
                ofs << particles_flat.at(iparticle * ndim + idim) << " ";
            }
            ofs << likelihood_ls.at(iparticle);
            ofs << std::endl;
        }
    }

    double gamma = 0.;
    int iter = 1;
    std::vector<double> cov_flat(ndim * ndim);
    std::vector<int> assigned_num;
    std::vector<int> work_assigned_num(work_size);
    std::vector<double> work_particles_flat_new;
    std::vector<double> work_likelihood_ls_new;
    std::vector<int> sum_assigned_ls(numprocs);
    std::vector<std::vector<double>> buf_likelihood(numprocs);
    std::vector<std::vector<double>> buf_particles_flat(numprocs);
    double st_time, en_time;
    while (1. - gamma > pow(10, -8)) {
        if (myid == 0) {
            st_time = MPI_Wtime();
            // list for resampling weights
            std::vector<double> weights(nparticle, 1.);
            // find the gamma such that c.o.v of weights = 0.5
            gamma = find_next_gamma(gamma, likelihood_ls, weights);
            std::cout << "gamma: " << gamma << std::endl;

            // normalize weights (sum of weights needs to be 1)
            weights = normalize_weights(weights);

            // calculate mean and covariance of the samples
            std::vector<double> mean =
                calc_mean_particles(particles_flat, weights, nparticle, ndim);
            cov_flat = calc_cov_particles(particles_flat, weights, mean,
                                          nparticle, ndim);

            assigned_num = resample_particles(nparticle, weights);

            reorder_to_send(assigned_num, particles_flat, nparticle, ndim,
                            numprocs, work_size);
            for (int iproc = 0; iproc < numprocs; iproc++) {
                int sum_assigned = 0;
                for (int iparticle = iproc * work_size;
                     iparticle < (iproc + 1) * work_size; iparticle++) {
                    sum_assigned += assigned_num.at(iparticle);
                }
                sum_assigned_ls.at(iproc) = sum_assigned;
                buf_likelihood.at(iproc).resize(sum_assigned);
                buf_particles_flat.at(iproc).resize(sum_assigned * ndim);
            }
        }
        MPI_Bcast(&gamma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cov_flat[0], ndim * ndim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(&particles_flat[0], work_size * ndim, MPI_DOUBLE,
                    &work_particles_flat.at(0), work_size * ndim, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        MPI_Scatter(&assigned_num[0], work_size, MPI_INT,
                    &work_assigned_num.at(0), work_size, MPI_INT, 0,
                    MPI_COMM_WORLD);
        if (myid == 0) {
            en_time = MPI_Wtime();
            std::cout << "1part time: " << en_time - st_time << std::endl;
            st_time = MPI_Wtime();
        }
        work_mcmc_sampling(work_assigned_num, work_particles_flat,
                           work_particles_flat_new, work_likelihood_ls_new,
                           work_size, ndim, cov_flat, gamma, obs_points, dvec,
                           obs_unitvec, obs_sigma, nsar, ngnss, nparticle_slip,
                           max_slip, nxi, neta, myid);
        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            en_time = MPI_Wtime();
            std::cout << "2part time: " << en_time - st_time << std::endl;
            st_time = MPI_Wtime();
        }
        if (myid == 0) {
            for (int iproc = 1; iproc < numprocs; iproc++) {
                MPI_Recv(&buf_likelihood.at(iproc).at(0),
                         sum_assigned_ls.at(iproc), MPI_DOUBLE, iproc, 0,
                         MPI_COMM_WORLD, &status);
                MPI_Recv(&buf_particles_flat.at(iproc).at(0),
                         sum_assigned_ls.at(iproc) * ndim, MPI_DOUBLE, iproc, 1,
                         MPI_COMM_WORLD, &status);
                // std::cout << "iproc: " << iproc << " recv size "
                //           << sum_assigned_ls.at(iproc) << std::endl;
            }
            for (int iparticle = 0; iparticle < work_likelihood_ls_new.size();
                 iparticle++) {
                buf_likelihood.at(0).at(iparticle) =
                    work_likelihood_ls_new.at(iparticle);
                for (int idim = 0; idim < ndim; idim++) {
                    buf_particles_flat.at(0).at(iparticle * ndim + idim) =
                        work_particles_flat_new.at(iparticle * ndim + idim);
                }
            }
            int cnt = 0;
            for (int iproc = 0; iproc < numprocs; iproc++) {
                int size = sum_assigned_ls.at(iproc);
                for (int iparticle = 0; iparticle < size; iparticle++) {
                    likelihood_ls.at(cnt) =
                        buf_likelihood.at(iproc).at(iparticle);
                    for (int idim = 0; idim < ndim; idim++) {
                        particles_flat.at(cnt * ndim + idim) =
                            buf_particles_flat.at(iproc).at(iparticle * ndim +
                                                            idim);
                    }
                    cnt++;
                }
            }
        } else {
            int send_size = work_likelihood_ls_new.size();
            MPI_Send(&work_likelihood_ls_new.at(0), send_size, MPI_DOUBLE, 0, 0,
                     MPI_COMM_WORLD);
            MPI_Send(&work_particles_flat_new.at(0), send_size * ndim,
                     MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }

        if (myid == 0) {
            // output result of stage j
            std::ofstream ofs(output_dir + std::to_string(iter) + ".csv");
            for (int iparticle = 0; iparticle < nparticle; iparticle++) {
                for (int idim = 0; idim < range.size(); idim++) {
                    ofs << particles_flat.at(iparticle * ndim + idim) << " ";
                }
                ofs << likelihood_ls.at(iparticle);
                ofs << std::endl;
            }
            iter++;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            en_time = MPI_Wtime();
            std::cout << "3part time: " << en_time - st_time << std::endl;
        }
    }

    // // output MAP value
    // int map_id = 0.;
    // int map_val = likelihood_ls.at(map_id);
    // for (int iparticle = 0; iparticle < nparticle; iparticle++) {
    //     double val = likelihood_ls.at(iparticle);
    //     if (val < map_val) {
    //         map_id = iparticle;
    //         map_val = val;
    //     }
    // }
    // std::cout << "map_id " << map_id << std::endl;
    // std::cout << "map_val " << map_val << std::endl;
    // std::ofstream ofs_map(output_dir + "map.csv");
    // auto particle_map = particles.at(map_id);
    // for (int idim = 0; idim < range.size(); idim++) {
    //     ofs_map << particle_map.at(idim) << " ";
    // }
    // ofs_map << likelihood_ls.at(map_id);
    // ofs_map << std::endl;

    // calculate mean and covariance of the samples
    if (myid == 0) {
        std::vector<double> weights_fin(nparticle, 1. / nparticle);
        std::vector<double> mean =
            calc_mean_particles(particles_flat, weights_fin, nparticle, ndim);
        std::vector<double> cov_flat = calc_cov_particles(
            particles_flat, weights_fin, mean, nparticle, ndim);
    }
    return;
}  // namespace smc_fault
}  // namespace smc_fault
