#include "linalg.hpp"
namespace linalg {
void print_matrix_double(const std::vector<std::vector<double>> &mat) {
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat.at(0).size(); j++) {
            std::cout << mat.at(i).at(j) << " ";
        }
        std::cout << std::endl;
    }
    return;
}

void print_vector_double(const std::vector<double> &vec) {
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec.at(i) << " ";
    }
    std::cout << std::endl;
    return;
}

std::vector<double> matvec(const std::vector<std::vector<double>> &mat,
                           const std::vector<double> &vec) {
    std::vector<double> ret(mat.size());
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat.at(0).size(); j++) {
            ret.at(i) += mat.at(i).at(j) * vec.at(j);
        }
    }
    return ret;
}

std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>> &mat) {
    std::vector<std::vector<double>> ret(mat.at(0).size(),
                                         std::vector<double>(mat.size()));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat.at(0).size(); j++) {
            ret.at(j).at(i) = mat.at(i).at(j);
        }
    }
    return ret;
}

std::vector<std::vector<double>> matmat(
    const std::vector<std::vector<double>> &amat,
    const std::vector<std::vector<double>> &bmat) {
    std::vector<std::vector<double>> ret(
        amat.size(), std::vector<double>(bmat.at(0).size()));
    for (int i = 0; i < amat.size(); i++) {
        for (int j = 0; j < bmat.at(0).size(); j++) {
            for (int k = 0; k < amat.at(0).size(); k++) {
                ret.at(i).at(j) += amat.at(i).at(k) * bmat.at(k).at(j);
            }
        }
    }
    return ret;
}

std::vector<std::vector<double>> matinv(std::vector<std::vector<double>> mat) {
    int n = mat.size();
    for (int i = 0; i < n; i++) {
        if (mat.at(i).size() != n) {
            std::cout << "size error" << std::endl;
            return {{}};
        }
    }
    std::vector<std::vector<double>> inv(n, std::vector<double>(n));
    std::vector<std::vector<double>> sweep(n, std::vector<double>(2 * n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sweep.at(i).at(j) = mat.at(i).at(j);
        }
        sweep.at(i).at(n + i) = 1.;
    }

    for (int k = 0; k < n; k++) {
        double max = fabs(sweep.at(k).at(k));
        int max_i = k;

        for (int i = k + 1; i < n; i++) {
            if (fabs(sweep.at(i).at(k)) > max) {
                max = fabs(sweep.at(i).at(k));
                max_i = i;
            }
        }

        if (fabs(sweep.at(max_i).at(k)) <= pow(10., -8.)) {
            std::cout << "Singular Matrix" << std::endl;
            return {{}};
        }

        if (k != max_i) {
            for (int j = 0; j < n * 2; j++) {
                double tmp = sweep.at(max_i).at(j);
                sweep.at(max_i).at(j) = sweep.at(k).at(j);
                sweep.at(k).at(j) = tmp;
            }
        }

        double a = 1. / sweep.at(k).at(k);

        for (int j = 0; j < n * 2; j++) {
            sweep.at(k).at(j) *= a;
        }

        for (int i = 0; i < n; i++) {
            if (i == k) {
                continue;
            }

            a = -sweep.at(i).at(k);

            for (int j = 0; j < n * 2; j++) {
                sweep.at(i).at(j) += sweep.at(k).at(j) * a;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv.at(i).at(j) = sweep.at(i).at(n + j);
        }
    }

    return inv;
}

}  // namespace linalg
