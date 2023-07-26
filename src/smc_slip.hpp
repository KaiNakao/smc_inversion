#include <mkl.h>
#include <mkl_lapacke.h>
#include <omp.h>

#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>

#include "linalg.hpp"
namespace smc_slip {
double cdf_norm(double x, double mu, double sigma2);

double pdf_norm(double x, double mu, double sigma2);

double calc_likelihood(const std::vector<double> &svec,
                       const std::vector<double> &dvec,
                       const std::vector<double> &obs_sigma,
                       const std::vector<double> &sigma2_full,
                       const std::vector<double> &gmat_flat,
                       const double &log_sigma_sar2,
                       const double &log_sigma_gnss2, const int &nsar,
                       const int &ngnss, double &delta_norm);

double calc_prior(const std::vector<double> &svec, const double &log_alpha2,
                  const std::vector<int> &lmat_index,
                  const std::vector<double> &lmat_val);

void gen_init_particles(
    std::vector<std::vector<double>> &particles,
    std::vector<double> &likelihood_ls, std::vector<double> &prior_ls,
    const int &nparticle, const std::vector<double> &dvec,
    const std::vector<double> &obs_sigma,
    const std::vector<double> &sigma2_full,
    const std::vector<double> &gmat_flat, const double &log_sigma_sar2,
    const double &log_sigma_gnss2, const int &nsar, const int &ngnss,
    const double &log_alpha2, const std::vector<int> &lmat_index,
    const std::vector<double> &lmat_val,
    const std::vector<std::vector<double>> &llmat, const double &max_slip);

std::vector<double> calc_mean_std_vector(const std::vector<double> &vec);

double find_next_gamma(const double &gamma_prev,
                       std::vector<double> &likelihood_ls,
                       std::vector<double> &weights, double &neglog_evidence);

std::vector<double> normalize_weights(const std::vector<double> &weights);

std::vector<double> calc_mean_particles(
    const std::vector<std::vector<double>> &particles,
    const std::vector<double> &weights);

std::vector<double> calc_cov_particles(
    const std::vector<std::vector<double>> &particles,
    const std::vector<double> &weights, const std::vector<double> &mean);

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
    const std::vector<double> &lmat_val, const double &max_slip);

double smc_exec(std::vector<std::vector<double>> &particles,
                const std::string &output_dir, const int &nparticle,
                const std::vector<double> &dvec,
                const std::vector<double> &obs_sigma,
                const std::vector<double> &sigma2_full,
                const std::vector<std::vector<double>> &gmat,
                const double &log_sigma_sar2, const double &log_sigma_gnss2,
                const int &nsar, const int &ngnss, const double &log_alpha2,
                const std::vector<int> &lmat_index,
                const std::vector<double> &lmat_val,
                const std::vector<std::vector<double>> &llmat,
                const std::vector<int> &id_dof, const double &max_slip);
}  // namespace smc_slip
