## This file contains an inverse dynamics controller between two frames
## Author : Avadesh Meduri
## Date : 16/03/2021

import numpy as np
import pinocchio as pin

#from . robot_impedance_controller import RobotImpedanceController

class InverseDynamicsController():

    def __init__(self, robot, eff_arr):
        """
        Input:
            robot : robot object returned by pinocchio wrapper
            eff_arr : end effector name arr
        """

        self.pin_robot = robot.pin_robot
        self.nq = self.pin_robot.nq
        self.nv = self.pin_robot.nv
        self.eff_arr = eff_arr

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

        tau_id = self.compute_id_torques(des_q, des_v, des_a)
        tau_eff = np.zeros(self.nv)
        for j in range(len(self.eff_arr)):
            J = pin.computeFrameJacobian(self.pin_robot.model, self.pin_robot.data, des_q,\
                     self.pin_robot.model.getFrameId(self.eff_arr[j]), pin.LOCAL_WORLD_ALIGNED)
            tau_eff += np.matmul(J.T, np.hstack((fff[j*3:(j+1)*3], np.zeros(3))))

        tau = (tau_id - tau_eff)[6:]
        tau_gain = -self.kp*(np.subtract(q[7:], des_q[7:].T)) - self.kd*(np.subtract(dq[6:], des_v[6:].T))
        return tau + tau_gain.T