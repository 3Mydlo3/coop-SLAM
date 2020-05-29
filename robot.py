import numpy as np

from raycasting_grid_map import *
from frenet_optimal_trajectory import *
from params import XY_RES, YAW_RES, ROBOT_INIT_POSITIONS, ROBOT_COLORS

class Robot:
    def __init__(self, map_, id_):
        self.map = map_
        self.id = id_
        self.map.assign_robot(self)
        self.grid = None
        self.grid_blacklist = [] # For marking currently unreachable grids
        init_pos = ROBOT_INIT_POSITIONS[self.id]
        if init_pos is None:
            init_pos = np.squeeze(self.map.random_position())
        self.init_pos = np.array(init_pos)
        self.direction = 0
        self.color = ROBOT_COLORS[self.id]
        self.XY_RES = XY_RES
        self.YAW_RES = YAW_RES
        self.init_variables()
        self.error_count = 0 # Number of errors when trying to find way to destination
        # Start moving
        self.simulate()

    def init_variables(self):
        """Initializes robot variables for trajectories"""
        self.c_speed = 10.0 / 3.6  # current speed [m/s]
        self.c_d = 0.0  # current lateral position [m]
        self.c_d_d = 0.0  # current lateral speed [m/s]
        self.c_d_dd = 0.0  # current lateral acceleration [m/s]
        self.s0 = 0.0  # current course position

    def simulate(self):
        """
        Method handling whole simulation of robot in current frame
        """
        # Check grid status and obtain new if needed
        # If no grids to explore then just scan area around
        if not self.grid_ok():
            self.raycasting()
            return False
        # Now let's try to reach assigned waypoint
        try:
            # First check if we arrived and if yes let's create new course for path planning
            self.check_arrived()
            self.plan_path()
            self.calculate_direction()
            self.grid_blacklist = [] # Reset grid blacklist
            self.error_count = 0 # Number of errors when trying to find way to destination
        except (AttributeError, IndexError):
            # No path could be found
            self.error_count += 1
            print(f"Robot {self.id} | Errors: {self.error_count}")
            self.generate_target_course()
        self.raycasting()
        return True

    def explore_grid(self):
        """Creates waypoint inside currently explored grid"""
        [wx, wy] = self.get_position()
        uncovered_positions = self.grid.get_uncovered_positions()
        target = uncovered_positions[:, np.random.choice(np.shape(uncovered_positions)[1])]
        target = self.grid.grid_to_map(target)
        wx = np.append(wx, target[1])
        wy = np.append(wy, target[0])
        return wx, wy

    def find_nearest_grid(self):
        """Finds nearest free grid to current robot"""
        grids = self.map.find_discoverable_grids(blacklist=self.grid_blacklist)
        if grids == []:
            # No grids available
            return None
        grids_positions = [grid.get_position() for grid in grids]
        current_position = self.get_position()
        current_position = np.array([current_position[1], current_position[0]])
        difference = current_position - grids_positions
        distance = np.hypot(difference[:, 0], difference[:, 1])
        nearest_grids = np.where(distance == np.min(distance))
        return grids[np.random.choice(nearest_grids[0])]

    def find_new_grid(self):
        """Finds new grid for exploration"""
        self.unassign_robot()
        nearest_grid = self.find_nearest_grid()
        if nearest_grid is None:
            return None
        if nearest_grid.assign_robot(self):
            return nearest_grid
        else:
            return None

    def find_random_grid(self):
        """Find new random grid for exploration"""
        # Unassign robot from previous grid
        self.unassign_robot()
        grids = self.map.find_discoverable_grids(blacklist=self.grid_blacklist)
        return np.random.choice(grids)

    def generate_target_course(self, wx=None, wy=None):
        """Handles course generation between given waypoints"""
        if wx is None or wy is None:
            if self.error_count >= 10:
                print(f"Robot {self.id} | Cannot explore current grid further, changing grid.")
                self.error_count = 0 # Reset counter
                self.grid_blacklist.append(self.grid) # Add grid to blacklist
                self.grid = self.find_new_grid()
                if not self.grid_ok():
                    return False
            self.init_variables()
            # Generate waypoint on some position undiscovered in grid
            self.wx, self.wy = self.explore_grid()
        self.tx, self.ty, self.tyaw, self.tc, self.csp = generate_target_course(self.wx, self.wy)

    def grid_ok(self):
        """
        Method handles all grid checks,
        Returns true when everything is ok
        and robot can proceed to exploration
        """
        if self.grid is not None and self.grid.is_covered():
            # If grid if fully covered robot should proceed to new grid
            self.unassign_robot()
            print(f"Robot {self.id} | Finished grid exploration.")
        if self.grid is None:
            self.grid = self.find_new_grid()
            if self.grid is None:
                return False
            print(f"Robot {self.id} | Has new grid assigned.")
        return True

    def plan_path(self):
        """Plans path along current course"""
        mapped_objects = self.map.get_mapped_objects()
        path = frenet_optimal_planning(
        self.csp, self.s0, self.c_speed, self.c_d,
        self.c_d_d, self.c_d_dd, mapped_objects, map_size=self.map.get_size())

        # Update status
        self.s0 = path.s[1]
        self.c_d = path.d[1]
        self.c_d_d = path.d_d[1]
        self.c_d_dd = path.d_dd[1]
        self.c_speed = path.s_d[1]
        self.path = path

    def check_arrived(self):
        """Checks if arrived at destination"""
        current_position = self.get_position()
        try:
            if (self.map.check_covered(self.wx[-1], self.wy[-1]) or
                np.hypot(current_position[0] - self.tx[-1], current_position[1] - self.ty[-1]) <= 1.0):
                # Scanned waypoint position or arrived at the destination
                self.generate_target_course()
                return True
        except AttributeError:
            # No waypoints assigned
            return False
        return False

    def raycasting(self):
        """Handles scanning and returns list of detected objects"""
        position = self.get_position()
        objects = self.map.get_map_objects()
        self.pmap, self.minx, self.maxx, self.miny, self.maxy, self.detected_objects = generate_ray_casting_grid_map(
            ox=objects[:,0], oy=objects[:,1], xyreso=self.XY_RES, yawreso=self.YAW_RES,
            posx=position[0], posy=position[1])
        if not len(self.detected_objects) == 0:
            self.map.map_objects(self.detected_objects)
        return self.detected_objects

    def unassign_robot(self):
        """Tries to unassign robot from grid and vice versa"""
        try:
            self.grid.unassign_robot(self)
            self.grid = None
            return True
        except AttributeError:
            # Robot is not assigned to any grid
            return False

    def get_assigned_grid(self):
        """Returns currently assigned grid to robot"""
        return self.grid

    def get_color(self):
        return self.color

    def get_current_grid(self):
        return self.map.get_grid(self.get_position())

    def get_speed(self):
        return self.c_speed

    def get_tx(self):
        return self.tx

    def get_ty(self):
        return self.ty

    def get_path(self):
        return self.path

    def get_waypoints(self):
        return self.wx, self.wy

    def get_last_waypoint(self):
        """Returns coordinates of last waypoint"""
        return self.wx[-1], self.wy[-1]

    def get_position(self):
        """Returns current robot position"""
        try:
            return np.array([self.path.x[1], self.path.y[1]])
        except AttributeError:
            return self.init_pos

    def get_scan_data(self):
        """Returns scanning data"""
        return get_heatmap(self.pmap, self.minx, self.maxx, self.miny, self.maxy, self.XY_RES)

    def get_direction(self):
        """Returns current robot direction"""
        return self.direction

    def calculate_direction(self):
        """Calculates current robot direction"""
        path = self.path
        self.direction = -90 + np.rad2deg(np.arctan2(path.y[1] - path.y[0], path.x[1] - path.x[0]))
