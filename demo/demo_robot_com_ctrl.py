##################################################################################################################
## This file is the centroidal controller demo for a given robot (solo in this case)
#################################################################################################################
## Author: Avadesh Meduri
## Date: 9/12/2020
#################################################################################################################

import numpy as np
import time

from blmc_controllers.robot_impedance_controller import RobotImpedanceController
from blmc_controllers.robot_centroidal_controller import RobotCentroidalController
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from robot_properties_bolt.bolt_wrapper import BoltRobot, BoltConfig

# Create a Pybullet simulation environment
env = BulletEnvWithGround()
solo = True
bolt = False
# Create a robot instance. This initializes the simulator as well.
if solo:
    robot = env.add_robot(Solo12Robot)
    RobotConfig = Solo12Config
elif bolt:
    robot = env.add_robot(BoltRobot)
    RobotConfig = BoltConfig

tau = np.zeros(robot.nb_dof)

# Move the default position closer to the ground.
initial_configuration = [0., 0., 0.21, 0., 0., 0., 1.] + robot.nb_ee * [0., 0.9, -1.8]
Solo12Config.initial_configuration = initial_configuration

# # Reset the robot to some initial state.
q0 = np.matrix(Solo12Config.initial_configuration).T
dq0 = np.matrix(Solo12Config.initial_velocity).T
robot.reset_state(q0, dq0)

x_com = [0.0, 0.0, 0.25]
xd_com = [0.0, 0.0, 0.0]

x_ori = [0., 0., 0., 1.]
x_angvel = [0., 0., 0.]
cnt_array = robot.nb_ee*[1,]

# Impedance controller gains
kp = robot.nb_ee * [0., 0., 0.] # Disable for now
kd = robot.nb_ee * [0., 0., 0.]
x_des = robot.nb_ee* [0., 0., -0.25] # Desired leg length
xd_des = robot.nb_ee * [0. ,0., 0.]

## initialising controllers ############################

# config_file = "./solo_impedance.yaml"
config_file = RobotConfig.paths["imp_ctrl_yaml"]

robot_cent_ctrl = RobotCentroidalController(robot,
        mu=0.6, kc=[200,200,200], dc=[5,5,5], kb=[200,200,200], db=[1.,1.,1.])
robot_leg_ctrl = RobotImpedanceController(robot, config_file)

# # Run the simulator for 100 steps
for i in range(4000):
    # Step the simulator.
    env.step(sleep=True) # You can sleep here if you want to slow down the replay
    # Read the final state and forces after the stepping.
    f_real = np.zeros(12)
    rl_cnt, act_force = robot.get_force()
    if len(rl_cnt) > 0:
        for k  in range(len(rl_cnt)):
            if rl_cnt[k] == robot.pin_robot.model.getFrameId("FL_FOOT") - 1:
                f_real[0:3] = -act_force[k][0:3]
            if rl_cnt[k] == robot.pin_robot.model.getFrameId("FR_FOOT") - 1:
                f_real[3:6] = -act_force[k][0:3]
            if rl_cnt[k] == robot.pin_robot.model.getFrameId("HL_FOOT") - 1:
                f_real[6:9] = -act_force[k][0:3]
            if rl_cnt[k] == robot.pin_robot.model.getFrameId("HR_FOOT") - 1:
                f_real[9:12] = -act_force[k][0:3]
    # print(rl_cnt)
    q, dq = robot.get_state()
    print(dq[0:3],  np.sum(f_real[2::3]))
    # computing forces to be applied in the centroidal space
    w_com = robot_cent_ctrl.compute_com_wrench(q, dq, x_com, xd_com, x_ori, x_angvel)
    # distrubuting forces to the active end effectors
    F = robot_cent_ctrl.compute_force_qp(q, dq, cnt_array, w_com)
    # passing forces to the impedance controller
    tau = robot_leg_ctrl.return_joint_torques(q,dq,kp,kd,x_des,xd_des,F)
    # passing torques to the robot
    robot.send_joint_command(tau)