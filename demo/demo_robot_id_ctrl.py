## This is demo of the inverse dynamics controller
## Author : Avadesh Meduri
## Date : 16/03/2021

import numpy as np
import time
import pinocchio as pin

from blmc_controllers.robot_id_controller import InverseDynamicsController
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

## initialising controllers ############################
eff_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
robot_ctrl = InverseDynamicsController(robot, eff_arr)
robot_ctrl.set_gains(5.0, 0.1)

des_q = q0
des_v = dq0
des_a = dq0
des_a[2] = -9.81

F = np.zeros((3*robot.nb_ee, ))
F[2::3] = pin.computeTotalMass(robot.pin_robot.model)*9.81/4.0

# # Run the simulator for 100 steps
for i in range(4000):
    # Step the simulator.
    env.step(sleep=True) # You can sleep here if you want to slow down the replay
    # Read the final state and forces after the stepping.
    q, dq = robot.get_state()
    # passing forces to the impedance controller
    tau = robot_ctrl.id_joint_torques(q,dq,des_q, des_v, des_a, F)
    # passing torques to the robot
    robot.send_joint_command(tau)