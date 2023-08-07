#include "smc_slip.hpp"

#include "mpi.h"

namespace smc_slip {

double cdf_norm(double x, double mu, double sigma2) {
    // cumulative distribution function of normal distribution
    return (1. + erf((x - mu) / sqrt(2 * sigma2))) / 2.;
}

double pdf_norm(double x, double mu, double sigma2) {
    // probability distribution function of normal distribution
    return exp(-pow(x - mu, 2.) / (2. * sigma2)) / sqrt(2. * M_PI * sigma2);
}

double calc_likelihood(const std::vector<double> &svec,
                       const std::vector<double> &dvec,
                       const std::vector<double> &obs_sigma,
                       const std::vector<double> &sigma2_full,
                       const std::vector<double> &gmat_flat,
                       const double &log_sigma_sar2,
                       const double &log_sigma_gnss2, const int &nsar,
                       const int &ngnss, double &delta_norm,
                       std::vector<double> &gsvec) {
    for (int i = 0; i < gsvec.size(); i++) {
        gsvec[i] = 0.;
    }
    // delta norm is a measure for residual
    // double st_time, en_time;
    // st_time = MPI_Wtime();
    cblas_dgemv(CblasRowMajor, CblasNoTrans, dvec.size(), svec.size(), 1.,
                &gmat_flat[0], svec.size(), &svec[0], 1, 0., &gsvec[0], 1);
    // en_time = MPI_Wtime(); printf("dgemv etime: %f\n", en_time - st_time);

    delta_norm = 0.;
    double delta_loss = 0;
    for (int i = 0; i < dvec.size(); i++) {
        delta_norm +=
            pow(dvec.at(i) - gsvec.at(i), 2.) / pow(obs_sigma.at(i), 2);
        delta_loss +=
            pow(dvec.at(i) - gsvec.at(i), 2.) / (2. * sigma2_full.at(i));
    }
    return nsar * log_sigma_sar2 / 2. + 3. * ngnss / 2. * log_sigma_gnss2 +
           delta_loss;
}

double calc_prior(const std::vector<double> &svec, const double &log_alpha2,
                  const std::vector<int> &lmat_index,
                  const std::vector<double> &lmat_val,
                  std::vector<double> &lsvec) {
    for (int i = 0; i < lsvec.size(); i++) {
        lsvec[i] = 0.;
    }
    for (int i = 0; i < lsvec.size(); i++) {
        for (int j = 0; j < 5; j++) {
            lsvec[i] += lmat_val[5 * i + j] * svec[lmat_index[5 * i + j]];
        }
    }

    double prior = 0.;
    for (int i = 0; i < lsvec.size(); i++) {
        prior += pow(lsvec.at(i), 2.) / (2. * exp(log_alpha2));
    }

    return prior;
}

void gen_init_particles(
    std::vector<std::vector<double>> &particles,
    std::vector<double> &likelihood_ls, std::vector<double> &prior_ls,
    const int &nparticle, const std::vector<double> &dvec,
    const std::vector<double> &obs_sigma,
    const std::vector<double> &sigma2_full,
    const std::vector<double> &gmat_flat, const double &log_sigma_sar2,
    const double &log_sigma_gnss2, const int &nsar, const int &ngnss,
    const double &log_alpha2, const std::vector<int> &lmat_index,
    const std::vector<double> &lmat_val, const std::vector<double> &llmat_flat,
    const double &max_slip, std::vector<double> &gsvec,
    std::vector<double> &lsvec, const int &ndim) {
    std::random_device seed_gen;
    std::mt19937 engine(12345);
    particles.resize(nparticle);

    // Gibbs sampling from Truncated Multi Variate Normal distribution
    std::vector<double> yvec(ndim, 0);
    // obtain nparticle samples
    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        // component-wise update
        for (int idim = 0; idim < yvec.size(); idim++) {
            // mean and variance for conditional probability distribution
            double sigma2_i =
                pow(llmat_flat.at(idim * ndim + idim) / exp(log_alpha2), -1);
            double mu_i = 0;
            // loop for (jdim != idim)
            for (int jdim = 0; jdim < idim; jdim++) {
                mu_i -= llmat_flat.at(idim * ndim + jdim) / exp(log_alpha2) *
                        yvec.at(jdim) * sigma2_i;
            }
            for (int jdim = idim + 1; jdim < yvec.size(); jdim++) {
                mu_i -= llmat_flat.at(idim * ndim + jdim) / exp(log_alpha2) *
                        yvec.at(jdim) * sigma2_i;
            }

            // sampling x from conditional distribution
            std::normal_distribution<> dist(mu_i, sqrt(sigma2_i));
            double x = dist(engine);
            double fx = cdf_norm(x, mu_i, sigma2_i);
            double f1 = cdf_norm(max_slip, mu_i, sigma2_i);
            double f0 = cdf_norm(0, mu_i, sigma2_i);

            // solve F(y) = (F(1) - F(0)) F(x) + F(0) by Newton's method
            // where F(:) is CDF of normal distribution
            double y_i = mu_i;
            double err = 1.;
            double y_prev = y_i + err;
            while (err > pow(10, -8)) {
                y_i -= (cdf_norm(y_i, mu_i, sigma2_i) - ((f1 - f0) * fx + f0)) /
                       pdf_norm(y_i, mu_i, sigma2_i);
                err = fabs(y_i - y_prev);
                y_prev = y_i;
            }
            yvec.at(idim) = y_i;
        }
        particles.at(iparticle) = yvec;

        // calculate negative log likelihood and prior
        double delta_norm;
        likelihood_ls.at(iparticle) = calc_likelihood(
            yvec, dvec, obs_sigma, sigma2_full, gmat_flat, log_sigma_sar2,
            log_sigma_gnss2, nsar, ngnss, delta_norm, gsvec);
        prior_ls.at(iparticle) =
            calc_prior(yvec, log_alpha2, lmat_index, lmat_val, lsvec);
    }
}

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
                       std::vector<double> &weights, double &neglog_evidence) {
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

    // Calulate S_j (mean of the weight)
    double evidence = 0.;
    double diff_gamma = gamma - gamma_prev;
    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        double likelihood = likelihood_ls.at(iparticle);
        evidence += exp(-diff_gamma * likelihood);
    }
    evidence /= nparticle;
    neglog_evidence = -log(evidence);
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
    return cov_flat;
}

void resample_particles(
    std::vector<std::vector<double>> &particles,
    const std::vector<double> &weights, std::vector<double> &likelihood_ls,
    std::vector<double> &prior_ls, std::vector<double> &cov_flat,
    const double &gamma, const std::vector<double> &dvec,
    const std::vector<double> &obs_sigma,
    const std::vector<double> &sigma2_full,
    const std::vector<double> &gmat_flat, const double &log_sigma_sar2,
    const double &log_sigma_gnss2, const int &nsar, const int &ngnss,
    const double &log_alpha2, const std::vector<int> &lmat_index,
    const std::vector<double> &lmat_val, const double &max_slip,
    std::vector<double> &gsvec, std::vector<double> &lsvec) {
    std::random_device seed_gen;
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
    // list for negative log prior of the resampled particles
    std::vector<double> prior_ls_new(n_particle);

    // resampling
    // list for the indice of original particle of each resampled particle
    std::vector<double> resampled_idx(n_particle);
    double deno = n_particle;
    std::vector<double> uvec(n_particle);
    for (int n = 0; n < n_particle; n++) {
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
    for (int i_particle = 0; i_particle < n_particle; i_particle++) {
        if (assigned_id[i_particle].size() == 0) {
            // no need to start MCMC
            continue;
        }

        // ----_cur means current configuration
        std::vector<double> particle_cur = particles.at(i_particle);
        double likelihood_cur = likelihood_ls.at(i_particle);
        double prior_cur = prior_ls.at(i_particle);
        // negative log of posterior
        double post_cur = gamma * likelihood_cur + prior_cur;

        for (int j_particle : assigned_id[i_particle]) {
            // propose particle_cand
            std::vector<double> rand(ndim);
            for (int idim = 0; idim < ndim; idim++) {
                rand.at(idim) = dist_stnorm(engine);
            }
            cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                        ndim, &cov_flat[0], ndim, &rand[0], 1);
            std::vector<double> particle_cand(ndim);
            for (int idim = 0; idim < ndim; idim++) {
                particle_cand[idim] = particle_cur[idim] + rand[idim];
                // non negative constraints
                particle_cand[idim] = fmax(0, particle_cand[idim]);
                // max slip constraints
                particle_cand[idim] = fmin(max_slip, particle_cand[idim]);
            }

            // calculate negative log likelihood/prior/posterior
            // of the proposed configuration
            double delta_norm = 0.;
            double likelihood_cand =
                calc_likelihood(particle_cand, dvec, obs_sigma, sigma2_full,
                                gmat_flat, log_sigma_sar2, log_sigma_gnss2,
                                nsar, ngnss, delta_norm, gsvec);
            double prior_cand = calc_prior(particle_cand, log_alpha2,
                                           lmat_index, lmat_val, lsvec);
            double post_cand = gamma * likelihood_cand + prior_cand;

            // metropolis test
            double metropolis = dist_metropolis(engine);
            if (exp(post_cur - post_cand) > metropolis) {
                // accept
                particle_cur = particle_cand;
                likelihood_cur = likelihood_cand;
                post_cur = post_cand;
            } else {
                // reject
                ;
            }

            // save to new particle list
            particles_new.at(j_particle) = particle_cur;
            likelihood_ls_new.at(j_particle) = likelihood_cur;
            prior_ls_new.at(j_particle) = prior_cur;
        }
    }
    // update configurations
    particles = particles_new;
    likelihood_ls = likelihood_ls_new;
    prior_ls = prior_ls_new;

    return;
}

double smc_exec(std::vector<std::vector<double>> &particles,
                const int &nparticle, const std::vector<double> &dvec,
                const std::vector<double> &obs_sigma,
                const std::vector<double> &sigma2_full,
                const std::vector<std::vector<double>> &gmat,
                const double &log_sigma_sar2, const double &log_sigma_gnss2,
                const int &nsar, const int &ngnss, const double &log_alpha2,
                const std::vector<int> &lmat_index,
                const std::vector<double> &lmat_val,
                const std::vector<double> &llmat_flat,
                const std::vector<int> &id_dof, const double &max_slip,
                const int &flag_output, const std::string &output_path) {
    // greens function
    const int ndim = gmat.at(0).size();
    std::vector<double> gmat_flat(gmat.size() * gmat.at(0).size());
    for (int i = 0; i < gmat.size(); i++) {
        for (int j = 0; j < gmat.at(0).size(); j++) {
            gmat_flat.at(i * ndim + j) = gmat.at(i).at(j);
        }
    }
    // gmat * svec
    std::vector<double> gsvec(dvec.size());
    // lmat * svec
    std::vector<double> lsvec(lmat_index.size() / 5);
    // list for (negative log) likelihood for each particles
    std::vector<double> likelihood_ls(nparticle);
    // list for (negative log) prior for each particles
    std::vector<double> prior_ls(nparticle);
    // sampling from the prior distribution
    gen_init_particles(particles, likelihood_ls, prior_ls, nparticle, dvec,
                       obs_sigma, sigma2_full, gmat_flat, log_sigma_sar2,
                       log_sigma_gnss2, nsar, ngnss, log_alpha2, lmat_index,
                       lmat_val, llmat_flat, max_slip, gsvec, lsvec, ndim);

    // output result of stage 0 (disabled)
    // std::ofstream ofs(output_dir + std::to_string(0) + ".csv");
    // for (int iparticle = 0; iparticle < nparticle; iparticle++) {
    //     const std::vector<double> particle = particles.at(iparticle);
    //     std::vector<double> slip(2 * lmat.size());
    //     for (int idof = 0; idof < id_dof.size(); idof++) {
    //         int inode = id_dof.at(idof);
    //         slip.at(2 * inode) = particle.at(2 * idof);
    //         slip.at(2 * inode + 1) = particle.at(2 * idof + 1);
    //     }
    //     for (int i = 0; i < slip.size(); i++) {
    //         ofs << slip.at(i) << " ";
    //     }
    //     ofs << likelihood_ls.at(iparticle);
    //     ofs << std::endl;
    // }

    // list for resampling weights
    std::vector<double> weights(nparticle);
    // list for S_j for each stage j
    // (prod of S_j is the estimated value of Integral)
    std::vector<double> neglog_evidence_vec;

    double gamma = 0.;
    int iter = 0;
    while (1. - gamma > pow(10, -8)) {
        // S_j
        double neglog_evidence;

        // find the gamma such that c.o.v of weights = 0.5
        gamma = find_next_gamma(gamma, likelihood_ls, weights, neglog_evidence);
        neglog_evidence_vec.push_back(neglog_evidence);

        // normalize weights (sum of weights needs to be 1)
        weights = normalize_weights(weights);

        // calculate mean and covariance of the samples
        std::vector<double> mean = calc_mean_particles(particles, weights);
        std::vector<double> cov_flat =
            calc_cov_particles(particles, weights, mean);

        // resampling and MCMC sampling from the updated distribution
        resample_particles(particles, weights, likelihood_ls, prior_ls,
                           cov_flat, gamma, dvec, obs_sigma, sigma2_full,
                           gmat_flat, log_sigma_sar2, log_sigma_gnss2, nsar,
                           ngnss, log_alpha2, lmat_index, lmat_val, max_slip,
                           gsvec, lsvec);

        // output result of stage j(disabled)
        // std::ofstream ofs(output_dir + std::to_string(iter) + ".csv");
        // for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        //     const std::vector<double> particle = particles.at(iparticle);
        //     std::vector<double> slip(lmat_index.size() / 5);
        //     for (int idof = 0; idof < id_dof.size(); idof++) {
        //         int inode = id_dof.at(idof);
        //         slip.at(2 * inode) = particle.at(2 * idof);
        //         slip.at(2 * inode + 1) = particle.at(2 * idof + 1);
        //     }
        //     for (int i = 0; i < slip.size(); i++) {
        //         ofs << slip.at(i) << " ";
        //     }
        //     ofs << likelihood_ls.at(iparticle) + prior_ls.at(iparticle);
        //     ofs << std::endl;
        // }
        iter++;
    }
    if (flag_output) {
        std::ofstream ofs(output_path);
        for (int iparticle = 0; iparticle < nparticle; iparticle++) {
            const std::vector<double> particle = particles.at(iparticle);
            std::vector<double> slip(lmat_index.size() / 5);
            for (int idof = 0; idof < id_dof.size(); idof++) {
                int inode = id_dof.at(idof);
                slip.at(2 * inode) = particle.at(2 * idof);
                slip.at(2 * inode + 1) = particle.at(2 * idof + 1);
            }
            for (int i = 0; i < slip.size(); i++) {
                ofs << slip.at(i) << " ";
            }
            ofs << likelihood_ls.at(iparticle) + prior_ls.at(iparticle);
            // ofs << std::endl;
            ofs << " ";
            double delta_norm = 0.;
            calc_likelihood(particle, dvec, obs_sigma, sigma2_full, gmat_flat,
                            log_sigma_sar2, log_sigma_gnss2, nsar, ngnss,
                            delta_norm, gsvec);
            ofs << delta_norm << std::endl;
        }
    }

    // product of S_j (sum for negative log value)
    double neglog_sum = 0.;
    for (int i = 0; i < neglog_evidence_vec.size(); i++) {
        neglog_sum += neglog_evidence_vec.at(i);
    }
    neglog_sum = fmin(pow(10, 10), neglog_sum);

    return neglog_sum;
}
}  // namespace smc_slip
