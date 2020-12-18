/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Python bindings for the StepperHead class
 */

#include "blmc_controllers/impedance_controller.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace blmc_controllers{

void bind_impedance_controller(py::module& module)
{
    py::class_<ImpedanceController>(module, "ImpedanceController")
        .def(py::init<>())

        // Public methods.
        .def("initialize",
             [](ImpedanceController& obj,
                py::object model,
                const std::string& root_frame_name,
                const std::string& end_frame_name)
             {
                 const pinocchio::Model& pinocchio_model =
                    model.cast<const pinocchio::Model&>();
                 obj.initialize(pinocchio_model, root_frame_name, end_frame_name);
                 return ;
             })
        .def("run", [](ImpedanceController& obj,
                       const Eigen::VectorXd& robot_configuration,
                       const Eigen::VectorXd& robot_velocity,
                       const ImpedanceController::Array6d& gain_proportional,
                       const ImpedanceController::Array6d& gain_derivative,
                       const double& gain_feed_forward_force,
                       py::object py_desired_end_frame_placement,
                       py::object py_desired_end_frame_velocity,
                       py::object py_feed_forward_force)
                    {
                        const pinocchio::SE3& desired_end_frame_placement = 
                            py_desired_end_frame_placement.cast<const pinocchio::SE3&>();
                        const pinocchio::Motion& desired_end_frame_velocity = 
                            py_desired_end_frame_velocity.cast<const pinocchio::Motion&>();
                        const pinocchio::Force& feed_forward_force = 
                            py_feed_forward_force.cast<const pinocchio::Force&>();
                        obj.run(robot_configuration, robot_velocity,
                                gain_proportional, gain_derivative,
                                gain_feed_forward_force,
                                desired_end_frame_placement,
                                desired_end_frame_velocity,
                                feed_forward_force);
                        return;
                    })
        .def("get_torques",
             &ImpedanceController::get_torques,
             py::return_value_policy::reference)
        .def("get_impedance_force",
             &ImpedanceController::get_impedance_force,
             py::return_value_policy::reference);
}

} // blmc_controllers