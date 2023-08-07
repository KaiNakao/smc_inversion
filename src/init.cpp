#include "init.hpp"
namespace init {
void discretize_fault(const double &lxi, const double &leta, const int &nxi,
                      const int &neta, std::vector<std::vector<int>> &cny,
                      std::vector<std::vector<double>> &coor,
                      std::unordered_map<int, std::vector<int>> &node_to_elem,
                      std::vector<int> &id_dof) {
    // num of patches = nxi * neta
    cny.resize(nxi * neta);
    // num of nodes = (nxi + 1) * (neta + 1)
    coor.resize((nxi + 1) * (neta + 1));
    // length of a patch
    double dxi = lxi / nxi;
    double deta = leta / neta;

    // coordinate of nodes
    for (int i = 0; i < nxi + 1; i++) {
        for (int j = 0; j < neta + 1; j++) {
            int id = i + (nxi + 1) * j;

            double xi = i * dxi;
            double eta = j * deta;
            coor.at(id) = {xi, eta};

            node_to_elem[id] = {};
            // no degree of freedom on the edge of the fault
            if (i == 0 || i == nxi || j == 0 || j == neta) {
                continue;
            }
            id_dof.push_back(id);
        }
    }

    // node id of patches
    for (int i = 0; i < nxi; i++) {
        for (int j = 0; j < neta; j++) {
            int id = i + nxi * j;
            int node0 = i + (nxi + 1) * j;
            int node1 = node0 + 1;
            int node2 = node1 + nxi + 1;
            int node3 = node0 + nxi + 1;
            cny.at(id) = {node0, node1, node2, node3};
            for (int node : cny.at(id)) {
                node_to_elem[node].push_back(id);
            }
        }
    }
    return;
}

void read_observation(const std::string &path,
                      std::vector<std::vector<double>> &obs_points,
                      std::vector<std::vector<double>> &obs_unitvec,
                      std::vector<double> &obs_sigma, std::vector<double> &dvec,
                      int &nsar, int &ngnss) {
    nsar = 0;
    ngnss = 0;
    std::ifstream ifs(path);
    std::string record;
    getline(ifs, record);  // header
    while (getline(ifs, record)) {
        std::vector<double> row;  // row = [x,y,ex,ey,ez,dlos,sigma,type]
        std::istringstream iss(record);
        for (int i = 0; i < 7; i++) {
            getline(iss, record, ',');
            row.push_back(std::stod(record));
        }
        getline(iss, record, ',');
        std::string type = record;
        if (type == "sar") {
            nsar++;
        }
        if (type == "gnss") {
            ngnss++;
        }
        obs_points.push_back({row.at(0), row.at(1)});
        obs_unitvec.push_back({row.at(2), row.at(3), row.at(4)});
        dvec.push_back(row.at(5));
        obs_sigma.push_back(row.at(6));
    }
    ngnss /= 3;  // GNSS obsevations have 3 direction components
}

std::vector<std::vector<double>> gen_laplacian(const int &nnode, const int &nxi,
                                               const int &neta,
                                               const double &dxi,
                                               const double &deta,
                                               const std::vector<int> &id_dof) {
    // laplacian for single component
    std::vector<std::vector<double>> luni(nnode, std::vector<double>(nnode));
    for (int inode = 0; inode < nnode; inode++) {
        if (inode % (nxi + 1) == 0) {
            luni.at(inode).at(inode + 1) += 2. / pow(dxi, 2.);
            luni.at(inode).at(inode) -= 2. / pow(dxi, 2.);
        } else if (inode % (nxi + 1) == nxi) {
            luni.at(inode).at(inode - 1) += 2. / (2 * pow(dxi, 2.));
            luni.at(inode).at(inode) -= 2. / (2 * pow(dxi, 2.));
        } else {
            luni.at(inode).at(inode - 1) += 1. / pow(dxi, 2.);
            luni.at(inode).at(inode + 1) += 1. / pow(dxi, 2.);
            luni.at(inode).at(inode) -= 2. / pow(dxi, 2.);
        }

        if (inode / (nxi + 1) == 0) {
            luni.at(inode).at(inode + (nxi + 1)) += 2. / pow(deta, 2.);
            luni.at(inode).at(inode) -= 2. / pow(deta, 2.);
        } else if (inode / (nxi + 1) == neta) {
            luni.at(inode).at(inode - (nxi + 1)) += 2. / pow(deta, 2.);
            luni.at(inode).at(inode) -= 2. / pow(deta, 2.);
        } else {
            luni.at(inode).at(inode - (nxi + 1)) += 1. / pow(deta, 2.);
            luni.at(inode).at(inode + (nxi + 1)) += 1. / pow(deta, 2.);
            luni.at(inode).at(inode) -= 2. / pow(deta, 2.);
        }
    }

    // laplacian for two components (u_xi, u_eta)
    std::vector<std::vector<double>> ret(
        2 * nnode, std::vector<double>(2 * id_dof.size()));
    for (int inode = 0; inode < nnode; inode++) {
        for (int idof = 0; idof < id_dof.size(); idof++) {
            int jnode = id_dof.at(idof);
            ret.at(2 * inode + 0).at(2 * idof + 0) = luni.at(inode).at(jnode);
            ret.at(2 * inode + 1).at(2 * idof + 1) = luni.at(inode).at(jnode);
        }
    }
    return ret;
}

std::vector<double> calc_ll(const std::vector<std::vector<double>> &lmat) {
    const int n = lmat.size();
    const int m = lmat.at(0).size();
    std::vector<double> lmat_flat(n * m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            lmat_flat.at(i * m + j) = lmat.at(i).at(j);
        }
    }
    std::vector<double> llmat_flat(m * m);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, m, n, 1.,
                &lmat_flat[0], m, &lmat_flat[0], m, 0., &llmat_flat[0], m);

    return llmat_flat;
}

void gen_sparse_lmat(const std::vector<std::vector<double>> &lmat,
                     std::vector<int> &lmat_index,
                     std::vector<double> &lmat_val) {
    lmat_index.resize(5 * lmat.size());
    lmat_val.resize(5 * lmat.size());
    for (int i = 0; i < lmat.size(); i++) {
        int cnt = 0;
        for (int j = 0; j < lmat.at(0).size(); j++) {
            double val = lmat.at(i).at(j);
            if (fabs(val) > pow(10, -8)) {
                // std::cout << j << " ";
                lmat_index[5 * i + cnt] = j;
                lmat_val[5 * i + cnt] = val;
                cnt++;
            }
        }
        while (cnt < 5) {
            lmat_index[5 * i + cnt] = 0;
            lmat_val[5 * i + cnt] = 0.;
            cnt++;
        }
    }
}

void read_observation_cv_train(const int &nfold, const int &cv_id,
                               std::vector<std::vector<double>> &obs_points,
                               std::vector<std::vector<double>> &obs_unitvec,
                               std::vector<double> &obs_sigma,
                               std::vector<double> &dvec, int &nsar,
                               int &ngnss) {
    nsar = 0;
    ngnss = 0;

    std::vector<std::vector<double>> obs_points_sar;
    std::vector<std::vector<double>> obs_unitvec_sar;
    std::vector<double> obs_sigma_sar;
    std::vector<double> dvec_sar;

    std::vector<std::vector<double>> obs_points_gnss;
    std::vector<std::vector<double>> obs_unitvec_gnss;
    std::vector<double> obs_sigma_gnss;
    std::vector<double> dvec_gnss;

    std::vector<int> train_batch_vec;
    for (int i = 0; i < nfold; i++) {
        if (i == cv_id) {
            continue;
        }
        train_batch_vec.push_back(i);
    }

    for (int batch_id : train_batch_vec) {
        std::string filename =
            "input_cv/observation_cv_" + std::to_string(batch_id) + ".csv";
        std::ifstream ifs(filename);
        std::string record;
        getline(ifs, record);  // header
        while (getline(ifs, record)) {
            std::vector<double> row;  // row = [x,y,ex,ey,ez,dlos,sigma,type]
            std::istringstream iss(record);
            for (int i = 0; i < 7; i++) {
                getline(iss, record, ',');
                row.push_back(std::stod(record));
            }
            getline(iss, record, ',');
            std::string type = record;
            if (type == "sar") {
                nsar++;
                obs_points_sar.push_back({row.at(0), row.at(1)});
                obs_unitvec_sar.push_back({row.at(2), row.at(3), row.at(4)});
                dvec_sar.push_back(row.at(5));
                obs_sigma_sar.push_back(row.at(6));
            }
            if (type == "gnss") {
                ngnss++;
                obs_points_gnss.push_back({row.at(0), row.at(1)});
                obs_unitvec_gnss.push_back({row.at(2), row.at(3), row.at(4)});
                dvec_gnss.push_back(row.at(5));
                obs_sigma_gnss.push_back(row.at(6));
            }
        }
    }

    obs_points.resize(0);
    obs_unitvec.resize(0);
    dvec.resize(0);
    obs_sigma.resize(0);

    for (int i = 0; i < nsar; i++) {
        obs_points.push_back(obs_points_sar.at(i));
        obs_unitvec.push_back(obs_unitvec_sar.at(i));
        dvec.push_back(dvec_sar.at(i));
        obs_sigma.push_back(obs_sigma_sar.at(i));
    }

    for (int i = 0; i < ngnss; i++) {
        obs_points.push_back(obs_points_gnss.at(i));
        obs_unitvec.push_back(obs_unitvec_gnss.at(i));
        dvec.push_back(dvec_gnss.at(i));
        obs_sigma.push_back(obs_sigma_gnss.at(i));
    }
    ngnss /= 3;
}

void read_observation_cv_valid(const int &nfold, const int &cv_id,
                               std::vector<std::vector<double>> &obs_points,
                               std::vector<std::vector<double>> &obs_unitvec,
                               std::vector<double> &obs_sigma,
                               std::vector<double> &dvec, int &nsar,
                               int &ngnss) {
    nsar = 0;
    ngnss = 0;

    std::vector<std::vector<double>> obs_points_sar;
    std::vector<std::vector<double>> obs_unitvec_sar;
    std::vector<double> obs_sigma_sar;
    std::vector<double> dvec_sar;

    std::vector<std::vector<double>> obs_points_gnss;
    std::vector<std::vector<double>> obs_unitvec_gnss;
    std::vector<double> obs_sigma_gnss;
    std::vector<double> dvec_gnss;

    std::vector<int> valid_batch_vec = {cv_id};
    for (int batch_id : valid_batch_vec) {
        std::string filename =
            "input_cv/observation_cv_" + std::to_string(batch_id) + ".csv";
        std::ifstream ifs(filename);
        std::string record;
        getline(ifs, record);  // header
        while (getline(ifs, record)) {
            std::vector<double> row;  // row = [x,y,ex,ey,ez,dlos,sigma,type]
            std::istringstream iss(record);
            for (int i = 0; i < 7; i++) {
                getline(iss, record, ',');
                row.push_back(std::stod(record));
            }
            getline(iss, record, ',');
            std::string type = record;
            if (type == "sar") {
                nsar++;
                obs_points_sar.push_back({row.at(0), row.at(1)});
                obs_unitvec_sar.push_back({row.at(2), row.at(3), row.at(4)});
                dvec_sar.push_back(row.at(5));
                obs_sigma_sar.push_back(row.at(6));
            }
            if (type == "gnss") {
                ngnss++;
                obs_points_gnss.push_back({row.at(0), row.at(1)});
                obs_unitvec_gnss.push_back({row.at(2), row.at(3), row.at(4)});
                dvec_gnss.push_back(row.at(5));
                obs_sigma_gnss.push_back(row.at(6));
            }
        }
    }

    obs_points.resize(0);
    obs_unitvec.resize(0);
    dvec.resize(0);
    obs_sigma.resize(0);

    for (int i = 0; i < nsar; i++) {
        obs_points.push_back(obs_points_sar.at(i));
        obs_unitvec.push_back(obs_unitvec_sar.at(i));
        dvec.push_back(dvec_sar.at(i));
        obs_sigma.push_back(obs_sigma_sar.at(i));
    }

    for (int i = 0; i < ngnss; i++) {
        obs_points.push_back(obs_points_gnss.at(i));
        obs_unitvec.push_back(obs_unitvec_gnss.at(i));
        dvec.push_back(dvec_gnss.at(i));
        obs_sigma.push_back(obs_sigma_gnss.at(i));
    }
    ngnss /= 3;
}
}  // namespace init
