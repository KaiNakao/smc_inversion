#include "smc_fault.hpp"

#include "mpi.h"

namespace smc_fault {

double calc_likelihood(
    const std::vector<double> &particle,
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<double> &dvec,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<std::vector<double>> &obs_unitvec,
    const std::vector<double> &obs_sigma, const double &leta,
    const std::unordered_map<int, std::vector<int>> &node_to_elem,
    const std::vector<int> &id_dof, const int &nsar, const int &ngnss,
    const std::vector<int> &lmat_index, const std::vector<double> &lmat_val,
    const std::vector<std::vector<double>> &llmat, const int nparticle_slip) {
    double xf = particle.at(0);
    double yf = particle.at(1);
    double zf = particle.at(2);
    double strike = particle.at(3);
    double dip = particle.at(4);
    double log_sigma_sar2 = particle.at(5);
    double log_sigma_gnss2 = particle.at(6);
    double log_alpha2 = particle.at(7);
    // Calculate greens function for the sampled fault
    auto gmat = gfunc::calc_greens_func(cny_fault, coor_fault, obs_points,
                                        obs_unitvec, leta, xf, yf, zf, strike,
                                        dip, node_to_elem, id_dof, nsar, ngnss);

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
    std::vector<std::vector<double>> particles_slip(nparticle_slip);
    double neglog = smc_slip::smc_exec(
        particles_slip, "output_slip/", nparticle_slip, dvec, obs_sigma,
        sigma2_full, gmat, log_sigma_sar2, log_sigma_gnss2, nsar, ngnss,
        log_alpha2, lmat_index, lmat_val, llmat, id_dof);
    return neglog;
}

std::vector<std::vector<double>> gen_init_particles(
    const int &nparticle, const std::vector<std::vector<double>> &range,
    std::vector<double> &likelihood_ls,
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<double> &dvec,
    const std::vector<std::vector<double>> &obs_unitvec,
    const std::vector<double> &obs_sigma, const double &leta,
    const std::unordered_map<int, std::vector<int>> &node_to_elem,
    const std::vector<int> &id_dof, const std::vector<int> &lmat_index,
    const std::vector<double> &lmat_val,
    const std::vector<std::vector<double>> &llmat, const int &nsar,
    const int &ngnss, const int &nparticle_slip, const int &myid,
    const int &numprocs) {
    // dimension of particle
    const int ndim = range.size();
    // samples to return
    std::vector<std::vector<double>> particles(nparticle,
                                               std::vector<double>(ndim));
    std::vector<double> particles_flat(nparticle * ndim);
    if (myid == 0) {
        // probability distribution instance for generating samples from piror
        // (uniform distribution)
        // std::random_device seed_gen;
        std::mt19937 engine(12345);
        std::vector<std::uniform_real_distribution<>> dist_vec(range.size());
        for (int idim = 0; idim < range.size(); idim++) {
            std::uniform_real_distribution<> dist(range.at(idim).at(0),
                                                  range.at(idim).at(1));
            dist_vec.at(idim) = dist;
        }

        for (int iparticle = 0; iparticle < nparticle; iparticle++) {
            // sampling
            for (int idim = 0; idim < range.size(); idim++) {
                double x = dist_vec.at(idim)(engine);
                particles_flat[iparticle * ndim + idim] = x;
                particles.at(iparticle).at(idim) = x;
            }
        }
    }
    MPI_Bcast(&particles_flat[0], nparticle * ndim, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    if (myid != 0) {
        for (int iparticle = 0; iparticle < nparticle; iparticle++) {
            for (int idim = 0; idim < range.size(); idim++) {
                particles.at(iparticle).at(idim) =
                    particles_flat[iparticle * ndim + idim];
            }
        }
    }
    int ib = (nparticle + (numprocs - 1)) / numprocs;
    int i_end = (myid + 1) * ib;
    if (i_end > nparticle) {
        i_end = nparticle;
    }
    std::vector<double> likelihood_ls_t(nparticle);
    // #pragma omp parallel for schedule(dynamic)
    for (int iparticle = myid * ib; iparticle < i_end; iparticle++) {
        std::vector<double> particle(ndim);
        for (int idim = 0; idim < ndim; idim++) {
            particle.at(idim) = particles_flat[iparticle * ndim + idim];
        }
        // calculate negative log likelihood for the sample
        likelihood_ls_t[iparticle] = calc_likelihood(
            particle, cny_fault, coor_fault, dvec, obs_points, obs_unitvec,
            obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss, lmat_index,
            lmat_val, llmat, nparticle_slip);
        std::cout << "iparticle: " << iparticle
                  << " likelihood: " << likelihood_ls_t[iparticle] << std::endl;
    }
    MPI_Reduce(&likelihood_ls_t[0], &likelihood_ls[0], nparticle, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    // if (myid == 0) {
    //     for (int i = 0; i < likelihood_ls.size(); i++) {
    //         std::cout << i << " " << likelihood_ls.at(i) << std::endl;
    //     }
    // }
    MPI_Barrier(MPI_COMM_WORLD);
    return particles;
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

        std::cout << "gamma: " << gamma << std::endl;
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
    const std::vector<std::vector<double>> &particles,
    const std::vector<double> &weights) {
    int n_particle = particles.size();
    int ndim = particles.at(0).size();
    std::vector<double> mean(ndim);
    for (int iparticle = 0; iparticle < n_particle; iparticle++) {
        for (int idim = 0; idim < ndim; idim++) {
            mean.at(idim) +=
                weights.at(iparticle) * particles.at(iparticle).at(idim);
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
    const std::vector<std::vector<double>> &particles,
    const std::vector<double> &weights, const std::vector<double> &mean) {
    int n_particle = particles.size();
    int ndim = particles.at(0).size();
    std::vector<double> cov_flat(ndim * ndim);
    for (int iparticle = 0; iparticle < n_particle; iparticle++) {
        const auto &particle = particles[iparticle];
        const auto &weight = weights[iparticle];
        for (int idim = 0; idim < ndim; idim++) {
            for (int jdim = 0; jdim < ndim; jdim++) {
                cov_flat[idim * ndim + jdim] += weight *
                                                (particle[idim] - mean[idim]) *
                                                (particle[jdim] - mean[jdim]);
            }
        }
    }
    for (int idim = 0; idim < ndim; idim++) {
        for (int jdim = 0; jdim < ndim; jdim++) {
            cov_flat[idim * ndim + jdim] *= 0.04;
        }
    }
    std::cout << "cov:" << std::endl;
    for (int i = 0; i < ndim; i++) {
        std::cout << cov_flat[i * ndim + i] << " ";
    }
    std::cout << std::endl;
    return cov_flat;
}

void resample_particles_parallel(
    std::vector<std::vector<double>> &particles,
    const std::vector<double> &weights, std::vector<double> &likelihood_ls,
    std::vector<double> &cov_flat, const double &gamma,
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<double> &dvec,
    const std::vector<std::vector<double>> &obs_unitvec,
    const std::vector<double> &obs_sigma, const double &leta,
    const std::unordered_map<int, std::vector<int>> &node_to_elem,
    const std::vector<int> &id_dof, const std::vector<int> &lmat_index,
    const std::vector<double> &lmat_val,
    const std::vector<std::vector<double>> &llmat, const int &nsar,
    const int &ngnss, const int &nparticle_slip, const int &myid,
    const int &numprocs) {
    // number of the samples
    const int n_particle = particles.size();
    // dimension of the samples
    const int ndim = particles.at(0).size();

    // list for resampled particles
    std::vector<double> particles_new_flat(n_particle * ndim);
    std::vector<double> particles_new_flat_t(n_particle * ndim);
    // list for negative log likelihood of the resampled particles
    std::vector<double> likelihood_ls_new(n_particle);
    std::vector<double> likelihood_ls_new_t(n_particle);

    // std::random_device seed_gen;
    std::mt19937 engine(12345);
    // probability distribution for MCCMC metropolis test
    std::uniform_real_distribution<> dist_metropolis(0., 1.);
    // standard normal distribution
    std::normal_distribution<> dist_stnorm(0., 1.);
    std::unordered_map<int, std::vector<int>> assigned_id;
    std::vector<int> assigned_id_flat;
    int max_len = 0;
    if (myid == 0) {
        // resampling
        // list for the indice of original particle of each resampled particle
        std::vector<int> resampled_idx(n_particle);
        double deno = n_particle;
        std::vector<double> uvec(n_particle);
        for (int n = 0; n < uvec.size(); n++) {
            std::uniform_real_distribution<> dist1(n / deno, (n + 1) / deno);
            uvec.at(n) = dist1(engine);
        }
        std::vector<double> cumsum(n_particle);
        cumsum.at(0) = weights.at(0);
        for (int n = 0; n < n_particle - 1; n++) {
            cumsum.at(n + 1) = cumsum.at(n) + weights.at(n + 1);
        }
        for (int i_particle = 0; i_particle < n_particle; i_particle++) {
            auto it = std::lower_bound(cumsum.begin(), cumsum.end(),
                                       uvec.at(i_particle));
            int i = std::distance(cumsum.begin(), it);
            resampled_idx.at(i_particle) = i;
        }

        // assigned_id[i_particle] =
        //  {original id of particles resampled to be i_particle}
        for (int i_particle = 0; i_particle < n_particle; i_particle++) {
            assigned_id[i_particle] = {};
        }
        for (int i_particle = 0; i_particle < n_particle; i_particle++) {
            int j_particle = resampled_idx.at(i_particle);
            assigned_id[j_particle].push_back(i_particle);
        }
        for (int i_particle = 0; i_particle < n_particle; i_particle++) {
            int len = assigned_id[i_particle].size();
            if (len > max_len) {
                max_len = len;
            }
        }
    }
    MPI_Bcast(&max_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    assigned_id_flat.resize(n_particle * max_len);
    for (int i = 0; i < n_particle * max_len; i++) {
        assigned_id_flat.at(i) = -1;
    }

    MPI_Status status;
    int iproc_size;
    std::vector<int> iprocs_ls;
    if (myid == 0) {
        std::vector<std::vector<int>> iprocs_send(numprocs);
        std::vector<int> range(n_particle);
        std::iota(range.begin(), range.end(), 0);
        std::sort(range.begin(), range.end(), [&](int a, int b) {
            return assigned_id.at(a).size() > assigned_id.at(b).size();
        });
        for (int i_particle = 0; i_particle < n_particle; i_particle++) {
            auto assigned = assigned_id.at(i_particle);
            for (int j = 0; j < assigned.size(); j++) {
                int j_particle = assigned.at(j);
                assigned_id_flat.at(i_particle * max_len + j) = j_particle;
            }
            int k_particle = range.at(i_particle);
            iprocs_send.at(i_particle % numprocs).push_back(k_particle);
        }
        for (int iproc = 1; iproc < numprocs; iproc++) {
            int size_send = iprocs_send.at(iproc).size();
            MPI_Send(&size_send, 1, MPI_INT, iproc, 0, MPI_COMM_WORLD);
            MPI_Send(&iprocs_send.at(iproc).at(0), iprocs_send.at(iproc).size(),
                     MPI_INT, iproc, 1, MPI_COMM_WORLD);
        }
        iproc_size = iprocs_send.at(0).size();
        iprocs_ls = iprocs_send.at(0);
    } else {
        MPI_Recv(&iproc_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        iprocs_ls.resize(iproc_size);
        MPI_Recv(&iprocs_ls.at(0), iproc_size, MPI_INT, 0, 1, MPI_COMM_WORLD,
                 &status);
        // std::cout << "myid: " << iprocs_ls.at(0) << " " << std::endl;
    }
    MPI_Bcast(&assigned_id_flat[0], n_particle * max_len, MPI_INT, 0,
              MPI_COMM_WORLD);
    // for (int i = 0; i < max_len * n_particle; i++) {
    //     std::cout << myid << " "
    //               << "index " << i << " " << assigned_id_flat[i] <<
    //               std::endl;
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(0);

    // LAPACK function for LU decomposition of matrix
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', ndim, &cov_flat[0], ndim);
    MPI_Barrier(MPI_COMM_WORLD);

    // MCMC sampling from the updated distribution
    // for (int i_particle = myid * ib; i_particle < i_end; i_particle++) {
    for (int i_particle : iprocs_ls) {
        // ----_cur means current configuration
        std::vector<double> particle_cur = particles.at(i_particle);
        double likelihood_cur;
        for (int j = 0; j < max_len; j++) {
            int j_particle = assigned_id_flat[i_particle * max_len + j];
            if (j_particle < 0) {
                break;
            }

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

            // calculate negative log likelihood of the candidate
            // configuration
            likelihood_cur = calc_likelihood(
                particle_cur, cny_fault, coor_fault, dvec, obs_points,
                obs_unitvec, obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss,
                lmat_index, lmat_val, llmat, nparticle_slip);

            // calculate negative log likelihood of the proposed
            // configuration
            double likelihood_cand = calc_likelihood(
                particle_cand, cny_fault, coor_fault, dvec, obs_points,
                obs_unitvec, obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss,
                lmat_index, lmat_val, llmat, nparticle_slip);
            double metropolis = dist_metropolis(engine);

            // metropolis test and check domain of definition
            if (particle_cand.at(4) < 90 && particle_cand.at(2) < 0 &&
                exp(gamma * (likelihood_cur - likelihood_cand)) > metropolis) {
                std::cout << "accepted likelihood: " << likelihood_cand
                          << std::endl;
                particle_cur = particle_cand;
                likelihood_cur = likelihood_cand;
            } else {
                std::cout << "rejected likelihood: " << likelihood_cand
                          << std::endl;
            }

            // save to new particle list
            // particles_new.at(j_particle) = particle_cur;
            for (int idim = 0; idim < ndim; idim++) {
                particles_new_flat_t[j_particle * ndim + idim] =
                    particle_cur[idim];
            }
            likelihood_ls_new_t.at(j_particle) = likelihood_cur;
        }
    }
    MPI_Allreduce(&particles_new_flat_t[0], &particles_new_flat[0],
                  ndim * n_particle, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Reduce(&likelihood_ls_new_t[0], &likelihood_ls_new[0], n_particle,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // update configurations
    if (myid == 0) {
        likelihood_ls = likelihood_ls_new;
    }
    for (int i_particle = 0; i_particle < n_particle; i_particle++) {
        for (int idim = 0; idim < ndim; idim++) {
            particles.at(i_particle).at(idim) =
                particles_new_flat.at(i_particle * ndim + idim);
        }
    }

    return;
}

void smc_exec(std::vector<std::vector<double>> &particles,
              const std::string &output_dir,
              const std::vector<std::vector<double>> &range,
              const int &nparticle,
              const std::vector<std::vector<int>> &cny_fault,
              const std::vector<std::vector<double>> &coor_fault,
              const std::vector<std::vector<double>> &obs_points,
              const std::vector<double> &dvec,
              const std::vector<std::vector<double>> &obs_unitvec,
              const std::vector<double> &obs_sigma, const double &leta,
              const std::unordered_map<int, std::vector<int>> &node_to_elem,
              const std::vector<int> &id_dof,
              const std::vector<int> &lmat_index,
              const std::vector<double> &lmat_val,
              const std::vector<std::vector<double>> &llmat, const int &nsar,
              const int &ngnss, const int &nparticle_slip, const int &myid,
              const int &numprocs) {
    const int ndim = range.size();
    std::cout << "myid : " << myid << " / " << numprocs << std::endl;
    // list for (negative log) likelihood for each particles
    std::vector<double> likelihood_ls(nparticle);
    // sampling from the prior distribution
    particles = gen_init_particles(
        nparticle, range, likelihood_ls, cny_fault, coor_fault, obs_points,
        dvec, obs_unitvec, obs_sigma, leta, node_to_elem, id_dof, lmat_index,
        lmat_val, llmat, nsar, ngnss, nparticle_slip, myid, numprocs);
    if (myid == 0) {
        // output result of stage 0
        std::ofstream ofs(output_dir + std::to_string(0) + ".csv");
        for (int iparticle = 0; iparticle < nparticle; iparticle++) {
            const std::vector<double> particle = particles.at(iparticle);
            for (int idim = 0; idim < range.size(); idim++) {
                ofs << particle.at(idim) << " ";
            }
            ofs << likelihood_ls.at(iparticle);
            ofs << std::endl;
        }
    }
    // barrier
    MPI_Barrier(MPI_COMM_WORLD);

    // list for resampling weights
    std::vector<double> weights(nparticle, 1.);
    double gamma = 0.;
    int iter = 1;
    std::vector<double> cov_flat(ndim * ndim);
    while (1. - gamma > pow(10, -8)) {
        if (myid == 0) {
            // find the gamma such that c.o.v of weights = 0.5
            gamma = find_next_gamma(gamma, likelihood_ls, weights);
            std::cout << "gamma: " << gamma << std::endl;

            // normalize weights (sum of weights needs to be 1)
            weights = normalize_weights(weights);

            // calculate mean and covariance of the samples
            std::vector<double> mean = calc_mean_particles(particles, weights);
            cov_flat = calc_cov_particles(particles, weights, mean);
        }
        MPI_Bcast(&gamma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cov_flat[0], ndim * ndim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // resampling and MCMC sampling from the updated distribution
        resample_particles_parallel(
            particles, weights, likelihood_ls, cov_flat, gamma, cny_fault,
            coor_fault, obs_points, dvec, obs_unitvec, obs_sigma, leta,
            node_to_elem, id_dof, lmat_index, lmat_val, llmat, nsar, ngnss,
            nparticle_slip, myid, numprocs);

        // output result of stage j
        if (myid == 0) {
            std::ofstream ofs(output_dir + std::to_string(iter) + ".csv");
            for (int iparticle = 0; iparticle < nparticle; iparticle++) {
                const std::vector<double> particle = particles.at(iparticle);
                for (int idim = 0; idim < range.size(); idim++) {
                    ofs << particle.at(idim) << " ";
                }
                ofs << likelihood_ls.at(iparticle);
                ofs << std::endl;
            }
        }
        iter++;
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
    return;
}
}  // namespace smc_fault
