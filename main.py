from raycasting_grid_map import *
from frenet_optimal_trajectory import *

SIM_TIME = 50.0 # simulation time [s]
# Map
MAP_SIZE = [100, 100] # X and Y size of map
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
        if init_pos is None:
            init_pos = np.squeeze(self.map.random_position())
        self.init_pos = init_pos
        self.color = color
        self.init_variables()
        # Start moving
        self.generate_target_course()

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
        try:
            self.plan_path()
        except (AttributeError, IndexError):
            self.generate_target_course()
        self.raycasting()
        self.check_arrived()

    def generate_waypoints(self):
        [wx, wy] = self.get_position()
        random_waypoints = np.random.randint(1, 4)
        w2 = self.map.random_position(count=random_waypoints)
        wx = np.append(wx, w2[:, 0])
        wy = np.append(wy, w2[:, 1])
        return wx, wy

    def generate_target_course(self, wx=None, wy=None):
        if wx is None or wy is None:
            self.init_variables()
            self.wx, self.wy = self.generate_waypoints()
        self.tx, self.ty, self.tyaw, self.tc, self.csp = generate_target_course(self.wx, self.wy)

    def plan_path(self):
        mapped_objects = self.map.get_mapped_objects()
        path = frenet_optimal_planning(
        self.csp, self.s0, self.c_speed, self.c_d, self.c_d_d, self.c_d_dd, mapped_objects)

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
        if np.hypot(current_position[0] - self.tx[-1], current_position[1] - self.ty[-1]) <= 1.0:
            self.generate_target_course()
            return True
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

    def get_color(self):
        return self.color

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

    def get_position(self):
        """Returns current robot position"""
        try:
            return (self.path.x[1], self.path.y[1])
        except AttributeError:
            return self.init_pos

    def get_scan_data(self):
        """Returns scanning data"""
        return get_heatmap(self.pmap, self.minx, self.maxx, self.miny, self.maxy, XY_RES)

    def get_direction(self):
        """Returns current robot direction"""
        path = self.path
        return -90 + np.rad2deg(np.arctan2(path.y[1] - path.y[0], path.x[1] - path.x[0]))


class Map:
    def __init__(self, map_size, objects_count):
        self.size = map_size
        self.objects = self._generate_objects(objects_count)
        self.mapped_objects = np.empty((0,2))
        self.robots = []
        self.map_coverage = self._prepare_map_coverage()

    def _generate_objects(self, objects_count):
        return self.random_position(objects_count)

    def _prepare_map_coverage(self):
        """Prepares 2-D array showing which map areas are already scanned"""
        map_coverage = np.zeros(shape=self.size)
        self.map_coverage_size = self.size[0] * self.size[1]
        return map_coverage

    def assign_robot(self, robot):
        """Assigns given robot to map"""
        self.robots.append(robot)

    def get_current_coverage(self):
        """Returns current map coverage in %"""
        return np.count_nonzero(self.map_coverage >= 0.5)/self.map_coverage_size * 100

    def get_map_objects(self):
        """Returns all objects on the map"""
        return self.objects

    def get_mapped_objects(self):
        """Returns all mapped objects"""
        return self.mapped_objects

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

def main():
    print(__file__ + " start!!")

    map_entity = Map(map_size=MAP_SIZE, objects_count=OBJECTS_COUNT)
    RFID = map_entity.get_map_objects()

    initial_positions = np.array([
        [0, 0],
        [0, 25],
        [0, 50],
        [0, 75]
    ])
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:yellow']
    # Create robots
    robots = []
    for _i in range(0, ROBOTS_COUNT):
        robots.append(Robot(map_=map_entity, init_pos=initial_positions[_i], color=colors[_i]))

    time = 0.0

    # Prepare plots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    while SIM_TIME >= time:
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
            ax1.set_title(f"Discorvered: {map_entity.get_current_coverage():.2f} %")
            ax1.set_aspect("equal")
            ax1.grid(True)
            # Plot stuff
            ax1.plot(RFID[:, 0], RFID[:, 1], "xk")
            for robot in robots:
                ax1.plot(robot.get_tx(), robot.get_ty())
                path = robot.get_path()
                ax1.plot(path.x[1:], path.y[1:], "-or")
                ax1.plot(*robot.get_position(), marker=(3, 0, robot.get_direction()), color=robot.get_color())
            map_entity.draw_current_scanning_area(ax=ax1)

            # Move map with followed robot
            path = robots[FOLLOW_ROBOT].get_path()
            ax1.set_xlim(path.x[1] - ANIM_AREA, path.x[1] + ANIM_AREA)
            ax1.set_ylim(path.y[1] - ANIM_AREA, path.y[1] + ANIM_AREA)

            # Draw raycasting
            ax1.plot(mapped_objects[:, 0], mapped_objects[:, 1], "xr")

            # Map plot
            ax2.set_title("Map")
            ax2.set_aspect("equal")
            ax2.grid(True)
            # Draw robots and waypoints
            for robot in robots:
                ax2.plot(*robot.get_position(), marker=(3, 0, robot.get_direction()), color=robot.get_color())
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
