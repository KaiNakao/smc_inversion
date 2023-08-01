#pragma once
#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "gfunc.hpp"
#include "init.hpp"
#include "smc_slip.hpp"

namespace smc_fault {

double calc_likelihood(const std::vector<double> &particle,
                       const std::vector<double> &dvec,
                       const std::vector<std::vector<double>> &obs_points,
                       const std::vector<std::vector<double>> &obs_unitvec,
                       const std::vector<double> &obs_sigma, const int &nsar,
                       const int &ngnss, const int nparticle_slip,
                       const double &max_slip, const int &nxi, const int &neta,
                       const int &flag_output, const std::string &output_path);

void sample_init_particles(std::vector<double> &particles_flat,
                           const int &nparticle, const int &ndim,
                           const std::vector<std::vector<double>> &range);

void work_eval_init_particles(
    const int &work_size, const int &ndim,
    const std::vector<double> &work_particles_flat,
    std::vector<double> &work_init_likelihood,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<double> &dvec,
    const std::vector<std::vector<double>> &obs_unitvec,
    const std::vector<double> &obs_sigma, const int &nsar, const int &ngnss,
    const int &nparticle_slip, const double &max_slip, const int &nxi,
    const int &neta, const int &myid);

std::vector<double> calc_mean_std_vector(const std::vector<double> &vec);

double find_next_gamma(const double &gamma_prev,
                       std::vector<double> &likelihood_ls,
                       std::vector<double> &weights);

std::vector<double> normalize_weights(const std::vector<double> &weights);

std::vector<double> calc_mean_particles(
    const std::vector<double> &particles_flat,
    const std::vector<double> &weights, const int &nparticle, const int &ndim);

std::vector<double> calc_cov_particles(
    const std::vector<double> &particles_flat,
    const std::vector<double> &weights, const std::vector<double> &mean,
    const int &nparticle, const int &ndim);

std::vector<int> resample_particles(const int &nparticle,
                                    const std::vector<double> &weights);

void reorder_to_send(std::vector<int> &assigned_num,
                     std::vector<double> &particles_flat, const int &nparticle,
                     const int &ndim, const int &numprocs,
                     const int &work_size);

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
                        const int &myid);

void obtain_final_slip_distribution(
    const std::vector<double> &particles_flat, const int &work_size,
    const int &ndim, std::vector<double> &work_particles_flat,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<double> &dvec,
    const std::vector<std::vector<double>> &obs_unitvec,
    const std::vector<double> &obs_sigma, const int &nsar, const int &ngnss,
    const int &nparticle_slip, const double &max_slip, const int &nxi,
    const int &neta, const int &myid, const std::string output_dir);

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
              const int &myid, const int &numprocs);
}  // namespace smc_fault
