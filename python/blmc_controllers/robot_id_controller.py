## This file contains an inverse dynamics controller between two frames
## Author : Avadesh Meduri
## Date : 16/03/2021

import numpy as np
import pinocchio as pin

from . robot_impedance_controller import RobotImpedanceController

class InverseDynamicsController(RobotImpedanceController):

    def __init__(self, robot, config_file):
        """
        Input:
            robot : robot object returned by pinocchio wrapper
            config_file : file that describes the desired frames to create springs in
        """
        super().__init__(robot, config_file)

        self.pin_robot = robot.pin_robot
        self.nq = self.pin_robot.nq
        self.nv = self.pin_robot.nv

    def set_gains(self, kp, kd):
        """
        This function is used to set the gains
        Input:
            kp : joint proportional gains
            kd : joint derivative gains
        """
        self.kp = kp
        self.kd = kd

    def compute_id_torques(self, q, v, a):
        """
        This function computes the torques for the give state using rnea
        Input:
            q : joint positions
            v : joint velocity
            a : joint acceleration
        """
        return np.reshape(pin.rnea(self.pin_robot.model, self.pin_robot.data, q, v, a), (self.nv,))

    def id_joint_torques(self, q, dq, des_q, des_v, des_a, fff):
        """
        This function computes the input torques with gains
        Input:
            q : joint positions
            dq : joint velocity
            des_q : desired joint positions
            des_v : desired joint velocities
            des_a : desired joint accelerations
            fff : desired feed forward force
        """
        assert len(q) == self.nq
        

        tau = self.compute_id_torques(des_q, des_v, des_a)[6:]
        tau_gain = -self.kp*(np.subtract(q[7:], des_q[7:].T)) - self.kd*(np.subtract(dq[6:], des_v[6:].T))
        fff_tau = self.return_joint_torques(q, dq, np.zeros(len(fff)), np.zeros(len(fff)),fff, fff, fff)
        fff_tau = np.reshape(fff_tau, (self.nv-6,))
        tau = np.add(tau, tau_gain).T

        return tau + fff_tau