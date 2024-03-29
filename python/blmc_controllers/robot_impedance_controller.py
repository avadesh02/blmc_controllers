##################################################################################################################
## This file creates impedance controllers between different frames 
## based on the input yaml file
#################################################################################################################
## Author: Avadesh Meduri & Paarth Shah
## Date: 9/12/2020
#################################################################################################################

import numpy as np
import yaml
from . impedance_controller import ImpedanceController
import pinocchio as pin

class RobotImpedanceController(ImpedanceController):

    def __init__(self, robot, config_file):
        '''
        Input:
            robot : robot object returned by pinocchio wrapper
            config_file : file that describes the desired frames to create springs in
        '''

        self.robot = robot
        self.num_eef = 0
        self.imp_ctrl_array = []
        self.initialise_impedance_controllers(config_file)

    def initialise_impedance_controllers(self, config_file):
        '''
        Reads the config file and initalises the impedance controllers
        Input:
            config_file : file that describes the desired frames to create springs in
        '''
        with open(config_file) as config:
            data_in = yaml.safe_load(config)
        for ctrls in data_in["impedance_controllers"]:
            if int(data_in["impedance_controllers"][ctrls]["is_eeff"]):
                self.num_eef += 1

            #TODO:  
            #Check here to make sure frame names exist in pinocchio robot_model/data structure

            #Append to list of impedance controllers
            self.imp_ctrl_array.append(ImpedanceController(ctrls, \
                                        self.robot, \
                                        data_in["impedance_controllers"][ctrls]["frame_root_name"], \
                                        data_in["impedance_controllers"][ctrls]["frame_end_name"], \
                                        int(data_in["impedance_controllers"][ctrls]["start_column"])
                                        ))
                        
    def return_joint_torques(self, q, dq, kp, kd, x_des, xd_des, f, grav_comp=True):
        '''
        Returns the joint torques at the current timestep
        Input:
            q : current joint positions
            dq : current joint velocities
            kp : Proportional gain
            kd : derivative gain
            x_des : desired lenghts with respect to the root frame for each controller (3*number_of_springs)
            xd_des : desired velocities with respect to the root frame
            f : feed forward forces
            grav_comp (boolean) : adds gravity compensation
        '''
        tau = np.zeros(len(self.imp_ctrl_array)*3)
        self.F_ = np.zeros(len(self.imp_ctrl_array)*3)

        for k in range(len(self.imp_ctrl_array)):
            s = slice(3*k, 3*(k+1))
            tau[s] = self.imp_ctrl_array[k].compute_impedance_torques(
                                    q,dq,
                                    kp[s],kd[s],x_des[s],
                                    xd_des[s],f[s])
            self.F_[s] = self.imp_ctrl_array[k].F_
        

        if grav_comp:
            tau += pin.rnea(self.robot.model, self.robot.data, q, dq, np.zeros(self.robot.model.nv))[6:]

        return tau

    def return_desired_forces(self):
        """
        This function returns the force total for all springs that is then computed
        to torques
        """
    
        return self.F_


    def return_joint_torques_world(self, q, dq, kp, kd, x_des, xd_des, f):
        '''
        Returns the joint torques at the current timestep, in world co-ordinates
        Input:
            q : current joint positions
            dq : current joint velocities
            kp : Proportional gain
            kd : derivative gain
            x_des : desired lenghts (3*number_of_springs)
            xd_des : desired velocities
            f : feed forward forces
        '''
        tau = np.zeros(len(self.imp_ctrl_array)*3)
        for k in range(len(self.imp_ctrl_array)):
            s = slice(3*k, 3*(k+1))
            tau[s] = self.imp_ctrl_array[k].compute_impedance_torques_world(
                                    q,dq,
                                    kp[s],kd[s],x_des[s],
                                    xd_des[s],f[s])
        
        return tau