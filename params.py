import numpy as np

SIM_TIME = 50.0 # simulation time [s]
DT = 0.2 # time step [s]
# Map
MAP_SIZE = [100, 100] # X and Y size of map
GRID_SIZE = 20 # Grid (square) size
DECAY_RATE = 0.002 # Rate of map info decay
OBJECTS_COUNT = 100 # Number of objects created on the map
# Raycasting
XY_RES = 0.25  # x-y grid resolution [m]
YAW_RES = np.deg2rad(2.0)  # yaw angle resolution [rad]
# Robots
ROBOTS_COUNT = 3 # Number of scanning robots
FOLLOW_ROBOT = 0 # Index of followed robot
ROBOT_INIT_POSITIONS = np.array([
    [0, 0],
    [0, 25],
    [0, 50],
    [0, 75]
])
ROBOT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
# Plots
ANIM_AREA = 20.0  # animation area length [m]
SHOW_ANIMATION = True # draw plots
