## This file contains an inverse dynamics controller between two frames
## Author : Avadesh Meduri
## Date : 16/03/2021

import numpy as np
import pinocchio as pin

from . qp_solver import quadprog_solve_qp

arr = lambda a: np.array(a).reshape(-1)
mat = lambda a: np.matrix(a).reshape((-1, 1))

class InverseDynamicsController():

    def __init__(self, robot, eff_arr, real_robot = False):
        """
        Input:
            robot : robot object returned by pinocchio wrapper
            eff_arr : end effector name arr
            real_robot : bool true if controller running on real robot
        """

        self.pin_robot = robot
        self.robot_mass = pin.computeTotalMass(self.pin_robot.model)
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

    def id_joint_torques(self, q, dq, des_q, des_v, des_a, fff, cnt_array):
        """
        This function computes the input torques with gains
        Input:
            q : joint positions
            dq : joint velocity
            des_q : desired joint positions
            des_v : desired joint velocities
            des_a : desired joint accelerations
            fff : desired feed forward force
            cnt_array
        """
        assert len(q) == self.nq
        self.J_arr = []

        # w_com = self.compute_com_wrench(q, dq, des_q, des_v.copy())
        tau_id = self.compute_id_torques(des_q, des_v, des_a)

        ## creating QP matrices
        N = int(len(self.eff_arr))

        for j in range(N):
            self.J_arr.append(pin.computeFrameJacobian(self.pin_robot.model, self.pin_robot.data, des_q,\
                     self.pin_robot.model.getFrameId(self.eff_arr[j]), pin.LOCAL_WORLD_ALIGNED).T)
        
        tau_eff = np.zeros(self.nv)
        for j in range(N):
            tau_eff += np.matmul(self.J_arr[j], np.hstack((fff[j*3:(j+1)*3], np.zeros(3))))

        tau = (tau_id - tau_eff)[6:]

        tau_gain = -self.kp*(np.subtract(q[7:].T, des_q[7:].T)) - self.kd*(np.subtract(dq[6:].T, des_v[6:].T))

        return tau + tau_gain.T

