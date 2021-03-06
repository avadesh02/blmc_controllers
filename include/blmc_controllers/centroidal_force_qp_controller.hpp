/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implements a qp allocating forces to track a desired centroidal wrench.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <eiquadprog/eiquadprog-fast.hpp>

namespace blmc_controllers
{
typedef Eigen::Array<double, 6, 1> Array6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/**
 * @brief Impedance controller between any two frames of the robot.
 */
class CentroidalForceQPController
{
public:
    /**
     * @brief Construct a new ImpedanceController object.
     */
    CentroidalForceQPController();

    /**
     * @brief Initialize the internal data. None real-time safe method.
     *
     * @param number_endeffectors Maximum number of endeffectors in the problem.
     * @param friction_coeff Floor friction coefficient to use.
     * @param qp_penalty_lin The penalty weight for the linear wcom violation.
     * @param qp_penalty_ang The penalty weight for the angular wcom violation.
     */
    void initialize(int number_endeffectors, double friction_coeff,
                    double qp_penalty_lin, double qp_penalty_ang);

    /**
     * @brief Computes the centroidal wrench using a PD controller.
     * 
     * @param w_com The desired centroidal wrench to track.
     * @param relative_position_endeff The relative position of the endeffectors
     *     with respect to the center of mass.
     */
    void run(Eigen::Ref<const Vector6d> w_com,
             Eigen::Ref<const Eigen::VectorXd> relative_position_endeff,
             Eigen::Ref<const Eigen::VectorXd> cnt_array);

    /**
     * @brief Get the computed desired forces
     *
     * @return Eigen::VectorXd&
     */
    Eigen::VectorXd& get_forces();

private:  // attributes
    /** @brief Output forces */
    Eigen::VectorXd forces_;
    Eigen::VectorXd sol_;

    Eigen::MatrixXd hess_;
    Eigen::MatrixXd ce_;
    Eigen::MatrixXd ce_new_;
    Eigen::MatrixXd ci_;

    Eigen::VectorXd g0_;
    Eigen::VectorXd ci0_;

    int nb_eff_;

    double mu_;

    double qp_penalty_lin_;
    double qp_penalty_ang_;

    eiquadprog::solvers::EiquadprogFast qp_;

};

}  // namespace blmc_controllers
