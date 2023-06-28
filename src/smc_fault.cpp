#include "smc_fault.hpp"

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
    const int &ngnss, const int &nparticle_slip) {
    // dimension of particle
    const int ndim = range.size();

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

    // samples to return
    std::vector<std::vector<double>> particles(nparticle,
                                               std::vector<double>(ndim));
#pragma omp parallel for schedule(dynamic)
    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        // sampling
        std::vector<double> particle(ndim);
#pragma omp critical
        {
            for (int idim = 0; idim < range.size(); idim++) {
                double x = dist_vec.at(idim)(engine);
                particle.at(idim) = x;
            }
        }
        particles.at(iparticle) = particle;
        // calculate negative log likelihood for the sample
        likelihood_ls.at(iparticle) = calc_likelihood(
            particle, cny_fault, coor_fault, dvec, obs_points, obs_unitvec,
            obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss, lmat_index,
            lmat_val, llmat, nparticle_slip);
        std::cout << "iparticle: " << iparticle
                  << " likelihood: " << likelihood_ls.at(iparticle)
                  << std::endl;
    }
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
    // std::vector<std::vector<double>> cov(ndim, std::vector<double>(ndim));
    std::vector<double> cov_flat(ndim * ndim);
    for (int iparticle = 0; iparticle < n_particle; iparticle++) {
        for (int idim = 0; idim < ndim; idim++) {
            for (int jdim = 0; jdim < ndim; jdim++) {
                cov_flat.at(idim * ndim + jdim) +=
                    weights.at(iparticle) *
                    (particles.at(iparticle).at(idim) - mean.at(idim)) *
                    (particles.at(iparticle).at(jdim) - mean.at(jdim));
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
        std::cout << cov_flat.at(i * ndim + i) << " ";
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
    const int &ngnss, const int &nparticle_slip) {
    // std::random_device seed_gen;
    std::mt19937 engine(12345);
    // probability distribution for MCCMC metropolis test
    std::uniform_real_distribution<> dist_metropolis(0., 1.);
    // standard normal distribution
    std::normal_distribution<> dist_stnorm(0., 1.);

    // number of the samples
    const int n_particle = particles.size();
    // dimension of the samples
    const int ndim = particles.at(0).size();

    // list for resampled particles
    std::vector<std::vector<double>> particles_new(n_particle);
    // list for negative log likelihood of the resampled particles
    std::vector<double> likelihood_ls_new(n_particle);

    // resampling
    // list for the indice of original particle of each resampled particle
    std::vector<double> resampled_idx(n_particle);
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
        auto it =
            std::lower_bound(cumsum.begin(), cumsum.end(), uvec.at(i_particle));
        int i = std::distance(cumsum.begin(), it);
        resampled_idx.at(i_particle) = i;
    }

    // assigned_id[i_particle] =
    //  {original id of particles resampled to be i_particle}
    std::unordered_map<int, std::vector<int>> assigned_id;
    for (int i_particle = 0; i_particle < n_particle; i_particle++) {
        assigned_id[i_particle] = {};
    }
    for (int i_particle = 0; i_particle < n_particle; i_particle++) {
        int j_particle = resampled_idx.at(i_particle);
        assigned_id[j_particle].push_back(i_particle);
    }

    // LAPACK function for LU decomposition of matrix
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', ndim, &cov_flat[0], ndim);

    // MCMC sampling from the updated distribution
#pragma omp parallel for schedule(dynamic)
    for (int i_particle = 0; i_particle < n_particle; i_particle++) {
        // if (assigned_id[i_particle].size() == 0) {
        //     // no need to start MCMC
        //     continue;
        // }

        // ----_cur means current configuration
        std::vector<double> particle_cur = particles.at(i_particle);

        for (int j_particle : assigned_id[i_particle]) {
            double likelihood_cur = calc_likelihood(
                particle_cur, cny_fault, coor_fault, dvec, obs_points,
                obs_unitvec, obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss,
                lmat_index, lmat_val, llmat, nparticle_slip);
            // propose particle_cand
            std::vector<double> rand(ndim);
#pragma omp critical
            {
                for (int idim = 0; idim < ndim; idim++) {
                    rand.at(idim) = dist_stnorm(engine);
                }
            }
            std::vector<double> particle_cand(ndim);
            for (int idim = 0; idim < ndim; idim++) {
                particle_cand[idim] = particle_cur[idim];
                for (int jdim = 0; jdim < idim + 1; jdim++) {
                    particle_cand[idim] +=
                        cov_flat[idim * ndim + jdim] * rand[jdim];
                }
            }

            // calculate negative log likelihood of the proposed configuration
            double likelihood_cand = calc_likelihood(
                particle_cand, cny_fault, coor_fault, dvec, obs_points,
                obs_unitvec, obs_sigma, leta, node_to_elem, id_dof, nsar, ngnss,
                lmat_index, lmat_val, llmat, nparticle_slip);
            double metropolis = dist_metropolis(engine);

            // metropolis test and check domain of definition
            if (particle_cand.at(4) < 90 && particle_cand.at(2) < 0 &&
                particle_cand.at(7) < -2 &&
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
            particles_new.at(j_particle) = particle_cur;
            likelihood_ls_new.at(j_particle) = likelihood_cur;
            // likelihood_cur = calc_likelihood(
            //     particle_cur, cny_fault, coor_fault, dvec, obs_points,
            //     obs_unitvec, obs_sigma, leta, node_to_elem, id_dof, nsar,
            //     ngnss, lmat_index, lmat_val, llmat, nparticle_slip);
        }
    }
    // update configurations
    likelihood_ls = likelihood_ls_new;
    particles = particles_new;

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
              const int &ngnss, const int &nparticle_slip) {
    const int ndim = range.size();
    // list for (negative log) likelihood for each particles
    std::vector<double> likelihood_ls(nparticle);
    // sampling from the prior distribution
    particles = gen_init_particles(
        nparticle, range, likelihood_ls, cny_fault, coor_fault, obs_points,
        dvec, obs_unitvec, obs_sigma, leta, node_to_elem, id_dof, lmat_index,
        lmat_val, llmat, nsar, ngnss, nparticle_slip);

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

    // list for resampling weights
    std::vector<double> weights(nparticle, 1.);
    double gamma = 0.;
    int iter = 1;
    while (1. - gamma > pow(10, -8)) {
        // find the gamma such that c.o.v of weights = 0.5
        gamma = find_next_gamma(gamma, likelihood_ls, weights);
        std::cout << "gamma: " << gamma << std::endl;

        // normalize weights (sum of weights needs to be 1)
        weights = normalize_weights(weights);

        // calculate mean and covariance of the samples
        std::vector<double> mean = calc_mean_particles(particles, weights);
        std::vector<double> cov_flat =
            calc_cov_particles(particles, weights, mean);

        // resampling and MCMC sampling from the updated distribution
        resample_particles_parallel(particles, weights, likelihood_ls, cov_flat,
                                    gamma, cny_fault, coor_fault, obs_points,
                                    dvec, obs_unitvec, obs_sigma, leta,
                                    node_to_elem, id_dof, lmat_index, lmat_val,
                                    llmat, nsar, ngnss, nparticle_slip);

        // output result of stage j
        std::ofstream ofs(output_dir + std::to_string(iter) + ".csv");
        for (int iparticle = 0; iparticle < nparticle; iparticle++) {
            const std::vector<double> particle = particles.at(iparticle);
            for (int idim = 0; idim < range.size(); idim++) {
                ofs << particle.at(idim) << " ";
            }
            ofs << likelihood_ls.at(iparticle);
            ofs << std::endl;
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

    // calculate mean and covariance of the samples
    std::vector<double> weights_fin(nparticle, 1. / nparticle);
    std::vector<double> mean = calc_mean_particles(particles, weights_fin);
    std::vector<double> cov_flat =
        calc_cov_particles(particles, weights_fin, mean);
    for (int i = 0; i < ndim; i++) {
        std::cout << cov_flat.at(i * ndim + i) / 0.04 << " ";
    }
    std::cout << std::endl;
    return;
}
}  // namespace smc_fault
