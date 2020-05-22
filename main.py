from raycasting_grid_map import *
from frenet_optimal_trajectory import *

SIM_TIME = 50.0 # simulation time [s]
# Map
MAP_SIZE = [100, 100] # X and Y size of map
GRID_SIZE = 20 # Grid (square) size
OBJECTS_COUNT = 100 # Number of objects created on the map
# Raycasting
XY_RES = 0.25  # x-y grid resolution [m]
YAW_RES = np.deg2rad(2.0)  # yaw angle resolution [rad]
# Robots
ROBOTS_COUNT = 3 # Number of scanning robots
FOLLOW_ROBOT = 0 # Index of followed robot
# Plots
ANIM_AREA = 20.0  # animation area length [m]

class Robot:
    def __init__(self, map_, init_pos=None, color=None):
        self.map = map_
        self.map.assign_robot(self)
        self.grid = None
        self.grid_blacklist = [] # For marking currently unreachable grids
        if init_pos is None:
            init_pos = np.squeeze(self.map.random_position())
        self.init_pos = np.array(init_pos)
        self.direction = 0
        self.color = color
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
            print(f"Robot: {self.color} | Errors: {self.error_count}")
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
        if wx is None or wy is None:
            if self.error_count >= 10:
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
        if self.grid is None:
            self.grid = self.find_new_grid()
            if self.grid is None:
                return False
        return True

    def plan_path(self):
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
        position = self.get_position()
        objects = self.map.get_map_objects()
        self.pmap, self.minx, self.maxx, self.miny, self.maxy, self.detected_objects = generate_ray_casting_grid_map(
            ox=objects[:,0], oy=objects[:,1], xyreso=XY_RES, yawreso=YAW_RES,
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
        return get_heatmap(self.pmap, self.minx, self.maxx, self.miny, self.maxy, XY_RES)

    def get_direction(self):
        """Returns current robot direction"""
        return self.direction

    def calculate_direction(self):
        """Calculates current robot direction"""
        path = self.path
        self.direction = -90 + np.rad2deg(np.arctan2(path.y[1] - path.y[0], path.x[1] - path.x[0]))


class Map:
    def __init__(self, map_size, grid_size, objects_count):
        self.size = np.array(map_size)
        self.objects = self._generate_objects(objects_count)
        self.mapped_objects = np.empty((0,2))
        self.robots = []
        self.map_coverage = self._prepare_map_coverage()
        self.grids = self._prepare_grids(grid_size)

    def _generate_objects(self, objects_count):
        return self.random_position(objects_count)

    def _prepare_grids(self, grid_size):
        """Prepares grids sized 20x20"""
        grids_count = (self.size / grid_size).astype(int)
        grids = []
        for i in range(0, grids_count[0]):
            for j in range(0, grids_count[1]):
                grids.append(Grid(map=self, grid_size=grid_size, coordinates=[i, j]))
        return grids

    def _prepare_map_coverage(self):
        """Prepares 2-D array showing which map areas are already scanned"""
        map_coverage = np.zeros(shape=self.size)
        self.map_coverage_size = self.size[0] * self.size[1]
        return map_coverage

    def assign_robot(self, robot):
        """Assigns given robot to map"""
        self.robots.append(robot)

    def find_discoverable_grids(self, blacklist=[]):
        discoverable_grids = []
        for grid in self.grids:
            if not grid in blacklist \
            and not grid.is_covered() \
            and grid.get_assigned_robot() is None:
                discoverable_grids.append(grid)
        return discoverable_grids

    def check_covered(self, x, y):
        """Checks if given position is already discovered"""
        try:
            return self.map_coverage[int(y), int(x)] > 0.01
        except IndexError:
            return False

    def get_current_coverage(self):
        """Returns current map coverage in %"""
        return np.count_nonzero(self.map_coverage >= 0.5)/self.map_coverage_size * 100

    def get_grid(self, position):
        grid_coordinates = np.floor(position / 10)
        return self.grids[grid_coordinates]

    def get_map_objects(self):
        """Returns all objects on the map"""
        return self.objects

    def get_mapped_objects(self):
        """Returns all mapped objects"""
        return self.mapped_objects

    def get_size(self):
        """Returns map size"""
        return self.size

    def map_objects(self, objects):
        """Tries to map given objects"""
        self.mapped_objects = np.unique(np.append(self.mapped_objects, objects, axis=0), axis=0)

    def random_position(self, count=None):
        """
        Returns random point within the map limits.
        Can return multiple points as one array with count parameter."""
        if count is None:
            count = 1
        return np.random.random(size=(count, 2)) * self.size

    def in_map(self, positionX=0, positionY=0):
        """Checks if given position is within map limits"""
        return (positionX >= 0 and positionX <= self.size[0]) and (positionY >= 0 and positionY <= self.size[1])

    def draw_current_scanning_area(self, ax=plt):
        """
        Draws current scanning data for all robots
        """
        # Create colormap
        cmap = plt.cm.Blues
        cmap.set_under(color='white')
        for robot in self.robots:
            # Get scan data for plot and map coverage update
            x, y, scan_data = robot.get_scan_data()
            self.update_map_coverage(x, y, scan_data)
            # Draw on plot
            ax.pcolormesh(x, y, scan_data, alpha=0.5, vmax=1.0, cmap=cmap, vmin=0.01)

    def update_map_coverage(self, x, y, scan_data):
        """
        Updates map coverage array according to scanner data
        """
        # Extract only coordinates vector
        x = x[:, 0]
        y = y[0, :]
        # Arange indexes of coordinates from vectors
        x_index = np.arange(0, len(x), 1)
        y_index = np.arange(0, len(y), 1)
        # Convert coordinates from vectors to map_coverage indexes
        x_map = [(int(x[i]), i) for i in x_index if self.in_map(positionX=x[i])]
        y_map = [(int(y[j]), j) for j in y_index if self.in_map(positionY=y[j])]
        for x_index, i in x_map:
            for y_index, j in y_map:
                # Check if grid is not already marked and grid was scanned now
                if self.map_coverage[y_index, x_index] < 0.5 and scan_data[i, j] >= 0.01:
                    self.map_coverage[y_index, x_index] = 0.5

    def draw_map_coverage(self, ax=plt):
        """Draws map coverage colormap on given plot"""
        # Create colormap
        cmap = plt.cm.Blues
        cmap.set_under(color='white')
        # Draw plot
        ax.pcolormesh(self.map_coverage, vmax=1.0, vmin=0.01, cmap=cmap)

class Grid:
    def __init__(self, map, grid_size, coordinates):
        """Stores data about grid's coverage and assignment"""
        self.map = map
        self.grid_coordinates = np.array(coordinates)
        self.grid_size = grid_size
        self.grid = self.calculate_grid()
        self.robot = None

    def assign_robot(self, robot):
        """Assigns given robot to discover this grid"""
        if self.robot is not None:
            return False
        self.robot = robot
        return True

    def unassign_robot(self, robot=None):
        """Unassigns given robot (default is assigned)"""
        if robot is not None:
            if self.robot == robot:
                self.robot = None
        else:
            self.robot = None

    def calculate_grid(self):
        self.shift = self.grid_coordinates * self.grid_size
        grid = self.map.map_coverage[self.shift[0] : self.shift[0] + self.grid_size, self.shift[1] : self.shift[1] + self.grid_size]
        return grid

    def is_covered(self):
        """Checks if grid is fully covered"""
        return np.all(self.grid >= 0.01)

    def get_assigned_robot(self):
        """Returns assigned robot"""
        return self.robot

    def get_position(self):
        """Returns center position of grid"""
        return self.grid_coordinates * self.grid_size + self.grid_size/2

    def get_uncovered_positions(self):
        """Returns all non-covered positions within the grid"""
        indexes = np.where(self.grid < 0.01)
        return np.vstack(indexes)

    def grid_to_map(self, position):
        """Transforms position from grid to map position"""
        return position + self.shift

def main():
    print(__file__ + " start!!")

    map_entity = Map(map_size=MAP_SIZE, grid_size=GRID_SIZE, objects_count=OBJECTS_COUNT)
    RFID = map_entity.get_map_objects()

    initial_positions = np.array([
        [0, 0],
        [0, 25],
        [0, 50],
        [0, 75]
    ])
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    # Create robots
    robots = []
    for _i in range(0, ROBOTS_COUNT):
        robots.append(Robot(map_=map_entity, init_pos=initial_positions[_i], color=colors[_i]))

    time = 0.0

    # Prepare plots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    while map_entity.get_current_coverage() < 100:
        # Frenet Optimal Trajectory
        for robot in robots:
            robot.simulate()

        time += DT

        if show_animation:  # pragma: no cover
            ax1.cla()
            ax2.cla()
            mapped_objects = map_entity.get_mapped_objects()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            # plot frenet
            ax1.set_title(f"Discovered: {map_entity.get_current_coverage():.2f} %")
            ax1.set_aspect("equal")
            ax1.grid(True)
            # Plot stuff
            ax1.plot(RFID[:, 0], RFID[:, 1], "xk")
            for robot in robots:
                try:
                    ax1.plot(robot.get_tx(), robot.get_ty())
                    path = robot.get_path()
                    ax1.plot(path.x[1:], path.y[1:], "-or")
                except AttributeError:
                    pass
                ax1.plot(*robot.get_position(), marker=(3, 0, robot.get_direction()), color=robot.get_color())
            map_entity.draw_current_scanning_area(ax=ax1)

            # Move map with followed robot
            robot_position = robots[FOLLOW_ROBOT].get_position()
            ax1.set_xlim(robot_position[0] - ANIM_AREA, robot_position[0] + ANIM_AREA)
            ax1.set_ylim(robot_position[1] - ANIM_AREA, robot_position[1] + ANIM_AREA)

            # Draw raycasting
            ax1.plot(mapped_objects[:, 0], mapped_objects[:, 1], "xr")

            # Map plot
            ax2.set_title("Map")
            ax2.set_aspect("equal")
            ax2.grid(True)
            # Draw robots and waypoints
            for robot in robots:
                ax2.plot(*robot.get_position(), marker=(3, 0, robot.get_direction()), color=robot.get_color())
                if robot.get_assigned_grid() is not None:
                    ax2.plot(*robot.get_waypoints(), "-or")
            # Draw mapped objects
            ax2.plot(mapped_objects[:, 0], mapped_objects[:, 1], "xr")
            ax2.set_xlim(0, MAP_SIZE[0])
            ax2.set_ylim(0, MAP_SIZE[1])
            map_entity.draw_map_coverage(ax=ax2)
            # Small pause
            plt.pause(0.001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

if __name__ == "__main__":
    main()
