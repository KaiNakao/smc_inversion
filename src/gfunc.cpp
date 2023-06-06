#include "gfunc.hpp"

#include <vector>

#include "linalg.hpp"

namespace gfunc {

std::vector<std::vector<double>> gen_unit_slip(
    const std::vector<std::vector<double>> &coor, const int &inode,
    const int &idirection) {
    // slip value at nodes (u_xi, u_eta components)
    std::vector<std::vector<double>> slip_dist(coor.size(),
                                               std::vector<double>(2));
    // in Iburi case, "the value of u_xi" is expected to be negative.
    // it need to be handled with positivity constraints
    if (idirection == 0) {
        slip_dist.at(inode).at(idirection) = -1.;
    }
    if (idirection == 1) {
        slip_dist.at(inode).at(idirection) = 1.;
    }
    // slip_dist.at(inode).at(idirection) = 1.;
    return slip_dist;
}

void call_dc3d0(const double &xsource, const double &ysource,
                const double &zsource, const double &xobs, const double &yobs,
                const double &uxi, const double &ueta, float &dip,
                const double &area, const double &strike,
                std::vector<double> &uret) {
    const double strike_rad = strike / 180. * M_PI;
    // in DC3D, x axis is the strike direction
    // rotation
    float x =
        sin(strike_rad) * (xobs - xsource) + cos(strike_rad) * (yobs - ysource);
    float y = -cos(strike_rad) * (xobs - xsource) +
              sin(strike_rad) * (yobs - ysource);
    float alpha = 2. / 3.;
    float z = 0.;
    float depth = -zsource;
    float pot1 = uxi * area;
    float pot2 = ueta * area;
    float pot3 = 0.;
    float pot4 = 0.;
    float ux;
    float uy;
    float uz;
    float uxx;
    float uyx;
    float uzx;
    float uxy;
    float uyy;
    float uzy;
    float uxz;
    float uyz;
    float uzz;
    int iret;
    dc3d0_(&alpha, &x, &y, &z, &depth, &dip, &pot1, &pot2, &pot3, &pot4, &ux,
           &uy, &uz, &uxx, &uyx, &uzx, &uxy, &uyy, &uzy, &uxz, &uyz, &uzz,
           &iret);
    // inverse rotation
    double ux_rot = sin(strike_rad) * ux - cos(strike_rad) * uy;
    double uy_rot = cos(strike_rad) * ux + sin(strike_rad) * uy;
    // adjust the unit of variables
    uret = {pow(10, 2) * ux_rot, pow(10, 2) * uy_rot, pow(10, 2) * uz};
    return;
}

std::vector<double> calc_responce(
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &slip_dist, const double &xobs,
    const double &yobs, const double &leta, const double &xf, const double &yf,
    const double &zf, const double &strike, const double dip,
    const std::vector<int> &target_ie) {
    // displacement (ux, uy, uz) at (xobs, yobs)
    std::vector<double> uret(3);
    // contribution from a point source
    std::vector<double> uobs(3);

    const double dip_rad = dip / 180. * M_PI;
    const double strike_rad = strike / 180. * M_PI;
    float dip_pass = dip;

    // loop for target patchs
    // (patchs containing the node)
    for (int ie : target_ie) {
        auto node_id = cny_fault.at(ie);

        // (xi, eta) and (u_xi, u_eta) at nodes
        std::vector<double> xinode(node_id.size());
        std::vector<double> etanode(node_id.size());
        std::vector<double> uxinode(node_id.size());
        std::vector<double> uetanode(node_id.size());
        for (int inode = 0; inode < node_id.size(); inode++) {
            xinode.at(inode) = coor_fault.at(node_id.at(inode)).at(0);
            etanode.at(inode) = coor_fault.at(node_id.at(inode)).at(1);
            uxinode.at(inode) = slip_dist.at(node_id.at(inode)).at(0);
            uetanode.at(inode) = slip_dist.at(node_id.at(inode)).at(1);
        }

        // surface area of the patch
        double area =
            (xinode.at(1) - xinode.at(0)) * (etanode.at(2) - etanode.at(1));

        // location of four point sources specified by
        // local coordinates (r1, r2)
        std::vector<double> r1vec = {-0.5, 0.5};
        std::vector<double> r2vec = {-0.5, 0.5};

        // loop for point sources
        for (double r1 : r1vec) {
            for (double r2 : r2vec) {
                // interpolate (xi, eta, uxi, ueta) by shape functions
                std::vector<double> nvec = {
                    (1. - r1) * (1 - r2) / 4.,
                    (1. + r1) * (1 - r2) / 4.,
                    (1. + r1) * (1 + r2) / 4.,
                    (1. - r1) * (1 + r2) / 4.,
                };
                double xi = 0.;
                double eta = 0.;
                double uxi = 0.;
                double ueta = 0.;
                for (int inode = 0; inode < node_id.size(); inode++) {
                    xi += nvec.at(inode) * xinode.at(inode);
                    eta += nvec.at(inode) * etanode.at(inode);
                    uxi += nvec.at(inode) * uxinode.at(inode);
                    ueta += nvec.at(inode) * uetanode.at(inode);
                }

                // location of the point source specified by
                // global coordinates (x, y, z)
                double xsource = xf +
                                 (leta - eta) * cos(dip_rad) * cos(strike_rad) +
                                 xi * sin(strike_rad);
                double ysource = yf -
                                 (leta - eta) * cos(dip_rad) * sin(strike_rad) +
                                 xi * cos(strike_rad);
                double zsource = zf - (leta - eta) * sin(dip_rad);

                // calculate displacement by Okada model
                call_dc3d0(xsource, ysource, zsource, xobs, yobs, uxi, ueta,
                           dip_pass, area / r1vec.size() / r2vec.size(), strike,
                           uret);

                // add contribution from the point source
                for (int idim = 0; idim < uobs.size(); idim++) {
                    uobs.at(idim) += uret.at(idim);
                }
            }
        }
    }
    return uobs;
}

std::vector<std::vector<double>> calc_responce_dist(
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &slip_dist, const double &leta,
    const double &xf, const double &yf, const double &zf, const double &strike,
    const double dip, const std::vector<int> &target_ie, const int &nsar,
    const int &ngnss) {
    // (ux, uy, uz) for all the observation points
    std::vector<std::vector<double>> ret(obs_points.size());
    // for SAR observation points
    for (int iobs = 0; iobs < nsar; iobs++) {
        double xobs = obs_points.at(iobs).at(0);
        double yobs = obs_points.at(iobs).at(1);
        // calculate displacement (ux, uy, uz) at single obsevation point
        auto uobs = calc_responce(cny_fault, coor_fault, slip_dist, xobs, yobs,
                                  leta, xf, yf, zf, strike, dip, target_ie);
        ret.at(iobs) = uobs;
    }
    // for GNSS observation points
    for (int iobs = 0; iobs < ngnss; iobs++) {
        double xobs = obs_points.at(nsar + 3 * iobs).at(0);
        double yobs = obs_points.at(nsar + 3 * iobs).at(1);
        // calculate displacement (ux, uy, uz) at single obsevation point
        auto uobs = calc_responce(cny_fault, coor_fault, slip_dist, xobs, yobs,
                                  leta, xf, yf, zf, strike, dip, target_ie);
        // copy for three components
        ret.at(nsar + 3 * iobs + 0) = uobs;
        ret.at(nsar + 3 * iobs + 1) = uobs;
        ret.at(nsar + 3 * iobs + 2) = uobs;
    }

    return ret;
}

std::vector<std::vector<double>> calc_greens_func(
    const std::vector<std::vector<int>> &cny_fault,
    const std::vector<std::vector<double>> &coor_fault,
    const std::vector<std::vector<double>> &obs_points,
    const std::vector<std::vector<double>> &obs_unitvec, const double &leta,
    const double &xf, const double &yf, const double &zf, const double &strike,
    const double &dip,
    const std::unordered_map<int, std::vector<int>> &node_to_elem,
    const std::vector<int> &id_dof, const int &nsar, const int &ngnss) {
    // matrix to return
    std::vector<std::vector<double>> gmat(
        obs_points.size(), std::vector<double>(2 * id_dof.size()));
    // loop for each degree of freedom of slip
    for (int idof = 0; idof < id_dof.size(); idof++) {
        int inode = id_dof.at(idof);
        for (int idirection = 0; idirection < 2; idirection++) {
            // slip distribution with single single unit slip
            auto slip_dist = gen_unit_slip(coor_fault, inode, idirection);
            // calculate displacement (x, y, z components) at all the
            // observation points
            auto response_dist = calc_responce_dist(
                obs_points, cny_fault, coor_fault, slip_dist, leta, xf, yf, zf,
                strike, dip, node_to_elem.at(inode), nsar, ngnss);
            // inner product (displacement * LOS unitvec)
            for (int iobs = 0; iobs < obs_points.size(); iobs++) {
                for (int idim = 0; idim < 3; idim++) {
                    gmat.at(iobs).at(2 * idof + idirection) +=
                        response_dist.at(iobs).at(idim) *
                        obs_unitvec.at(iobs).at(idim);
                }
            }
        }
    }
    return gmat;
}

}  // namespace gfunc
