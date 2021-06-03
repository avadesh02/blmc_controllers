## This file contains a TSID based Inverse Dynamics Controller
## Author : Avadesh Meduri
## Date : 16/03/2021

import numpy as np
import pinocchio as pin
import tsid
import time


class TSID_controller():

    def __init__(self, robot, urdf_path, model_path, eff_arr, q0, v0, mu=0.6):
        """
        Input:
            robot : robot object returned by pinocchio wrapper
            eff_arr : end effector name arr
        """
        self.kp_com = 2.0
        self.kp_contact = 1.0
        self.kp_posture = 50.0
        self.kp_orientation = 100.0

        self.w_com = 1e-5
        self.w_posture = 1.0e-6
        self.w_force = 1e1
        self.w_orientation = 1.0

        self.contact_transition = 0.1

        self.pin_robot = robot.pin_robot
        self.pin_data = robot.pin_data
        self.nq = self.pin_robot.nq
        self.nv = self.pin_robot.nv
        self.eff_arr = eff_arr
        self.curr_cnt_array = np.zeros(len(self.eff_arr))

        self.tsid_robot = tsid.RobotWrapper(urdf_path, [model_path], pin.JointModelFreeFlyer(), False)
        self.tsid_model = self.tsid_robot.model()
        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", self.tsid_robot, False)
        self.qp_solver = tsid.SolverHQuadProgFast("qp_solver")

        #Set up initial solve
        self.invdyn.computeProblemData(0, q0, v0)
        self.inv_dyn_data = self.invdyn.data()
        self.tsid_robot.computeAllTerms(self.inv_dyn_data, q0, v0)

        self.mu = mu

        # Add Contact Tasks (setup as equality constraints)
        # contactNormal = np.array([0., 0., 1.])
        # self.contact_arr = len(self.eff_arr)*[None]
        # for i, name in enumerate(self.eff_arr):
        #     print(name)
        #     self.contact_arr[i] = tsid.ContactPoint(name, self.tsid_robot, name, contactNormal, self.mu, 0.01, 25)
        #     self.contact_arr[i].setKp(self.kp_contact * np.ones(3))
        #     self.contact_arr[i].setKd(.03 * np.ones(3))
            #cnt_ref = self.tsid_robot.framePosition(self.inv_dyn_data, self.tsid_robot.model().getFrameId(name))
            #self.contact_arr[i].setReference(cnt_ref)
            # self.contact_arr[i].useLocalFrame(False)
            #self.invdyn.addRigidContact(self.contact_arr[i], self.w_force)

        # Add posture tasks (in the cost function)
        self.postureTask = tsid.TaskJointPosture("posture_task", self.tsid_robot)
        self.postureTask.setKp(self.kp_posture * np.ones(self.tsid_robot.nv-6))
        self.postureTask.setKd(2.0 * np.sqrt(self.kp_posture) * np.ones(3))
        self.invdyn.addMotionTask(self.postureTask, self.w_posture, 1, 0.0)

        #Add CoM task (in the cost function)
        # self.comTask = tsid.TaskComEquality("task-com", self.tsid_robot)
        # self.comTask.setKp(self.kp_com * np.ones(6))
        # self.comTask.setKd(.01 * np.ones(6))
        # self.invdyn.addMotionTask(self.comTask, self.w_com, 1, 0.0)

        #Setup trajectories (des_q, des_v, des_a)
        # self.com_ref = self.tsid_robot.com(self.inv_dyn_data)
        # self.trajCom = tsid.TrajectoryEuclidianConstant("trajectory_com", self.com_ref)
        # self.com_reference = self.trajCom.computeNext()

        self.trajPosture = tsid.TrajectoryEuclidianConstant("trajectory_posture", q0[7:])
        self.traj_reference = self.trajPosture.computeNext()

        self.qp_solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)

    def set_gains(self, kp_com, kp_contact, kp_posture, kp_orientation):
        self.kp_com = kp_com
        self.kp_contact = kp_contact
        self.kp_posture = kp_posture
        self.kp_orientation = kp_orientation

        # self.comTask.setKp(self.kp_com * np.ones(3))
        # self.comTask.setKd(2.0 * np.sqrt(self.kp_com) * np.ones(3))

        self.postureTask.setKp(self.kp_posture * np.ones(self.tsid_robot.nv-6))
        self.postureTask.setKd(2.0 * np.sqrt(self.kp_posture) * np.ones(3))

    def compute_id_torques(self, t, q, v, des_q, des_v, des_a, feed_fwd_forces, des_cnt_array):
        """
        This function computes the input torques with gains
        Input:
            q : joint positions
            v : joint velocity
            des_q : desired joint positions
            des_v : desired joint velocities
            des_a : desired joint accelerations
            fff : desired feed forward force
            des_cnt_array: desired binary contact array (as given by simulator/real contact detector)
        """
        assert len(q) == self.nq

        HQPData = self.invdyn.computeProblemData(t, q, v)
        
        #Update contacts of inverse dynamics array (i.e. which feet are and are not in contact):
        #TODO: make this list searchable via names rather than assuming the correct order
        # for i, name in enumerate(self.eff_arr):
        #     if self.curr_cnt_array[i] != des_cnt_array[i]:
        #         self.curr_cnt_array[i] = des_cnt_array[i].copy()
        #         if des_cnt_array[i] == 0:
        #             #self.invdyn.removeRigidContact(self.contact_arr[i].name, self.contact_transition)
        #             print("should remove contact")
        #             # print(t)
        #             # print(name)
        #             # time.sleep(3.0)
        #         else:
        #             # time.sleep(0.25)
        #             # print("Adding contact")
        #             # print(name)
        #             # print(t)
        #             self.inv_dyn_data = self.invdyn.data()
        #             self.tsid_robot.computeAllTerms(self.inv_dyn_data, q, v)
        #             cnt_ref = self.tsid_robot.framePosition(self.inv_dyn_data, self.tsid_robot.model().getFrameId(name))
        #             #self.contact_arr[i].setReference(cnt_ref)
        #             #self.contact_arr[i].setForceReference(feed_fwd_forces[i*3:i*3 + 3])
        #             self.invdyn.addRigidContact(self.contact_arr[i], self.w_force)

        #Set desired references
        # self.com_reference.pos(pin.centerOfMass(self.tsid_robot.model(), self.tsid_robot.data(), des_q, des_v))
        # self.com_reference.vel(des_v[0:3])
        # self.com_reference.acc(des_a[0:3])

        # self.traj_reference.pos(des_q)
        # self.traj_reference.vel(des_v)
        self.traj_reference.acc(des_a)

        #Need to update end-effector references here
        # self.comTask.setReference(self.com_reference)
        self.postureTask.setReference(self.traj_reference)
        # self.postureTask.compute(t, q, v, self.tsid_robot.data())

        #print(des_a[6:])
        #print(self.postureTask.getDesiredAcceleration)

        # for i in range(len(self.eff_arr)):
        #     if des_cnt_array[i] == 1:
        #         self.contact_arr[i].setForceReference(feed_fwd_forces[i*3:i*3 + 3])

        #Solve
        #print(feed_fwd_forces)
        sol = self.qp_solver.solve(HQPData)

        # print("Time %.3f"%(t))
        # print("\tNormal forces: ", end=' ')
        # for contact in self.contact_arr:
        #     if self.invdyn.checkContact(contact.name, sol):
        #         f = self.invdyn.getContactForce(contact.name, sol)
                # print("%4.2f"%(contact.getNormalForce(f)), end=' ')

        # if(sol.status!=0):
        #     print("[%d] QP problem could not be solved! Error code:"%(i), sol.status)

        tau = self.invdyn.getActuatorForces(sol)
        print(tau.shape)
        # tau_gain = -1.0*(np.subtract(q[7:], des_q[7:].T)) - 0.09*(np.subtract(v[6:], des_v[6:].T))
        # tau += tau_gain
        return tau