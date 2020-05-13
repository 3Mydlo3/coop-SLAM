from raycasting_grid_map import *
from frenet_optimal_trajectory import *

SIM_TIME = 50.0 # simulation time [s]
MAP_SIZE = [100, 100] # X and Y size of map
ROBOTS_COUNT = 3 # Number of scanning robots
FOLLOW_ROBOT = 0 # Index of followed robot

class Robot:
    def __init__(self, init_pos=None, color=None):
        if init_pos is None:
            init_pos = np.squeeze(random_on_map())
        self.color = color
        self.c_speed = 10.0 / 3.6  # current speed [m/s]
        self.c_d = init_pos[0]  # current lateral position [m]
        self.c_d_d = 0.0  # current lateral speed [m/s]
        self.c_d_dd = 0.0  # current lateral acceleration [m/s]
        self.s0 = init_pos[1]  # current course position

        # Start moving
        self.generate_target_course()

    def generate_waypoints(self):
        wx = [self.s0]
        wy = [self.c_d]
        random_waypoints = np.random.randint(1, 4)
        wx2 = np.random.random(size=(random_waypoints)) * MAP_SIZE[0]
        wy2 = np.random.random(size=(random_waypoints)) * MAP_SIZE[1]
        wx = np.append(wx, wx2)
        wy = np.append(wy, wy2)
        return wx, wy

    def generate_target_course(self, wx=None, wy=None):
        if wx is None or wy is None:
            self.wx, self.wy = self.generate_waypoints()
        self.tx, self.ty, self.tyaw, self.tc, self.csp = generate_target_course(self.wx, self.wy)

    def plan_path(self, mapped_objects):
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
        if np.hypot(self.path.x[1] - self.tx[-1], self.path.y[1] - self.ty[-1]) <= 1.0:
            self.generate_target_course()
            return True
        return False

    def raycasting(self, landmarksX, landmarksY, xyreso, yawreso):
        position = self.get_position()
        self.pmap, self.minx, self.maxx, self.miny, self.maxy, self.xyreso, self.detected_objects = generate_ray_casting_grid_map(
            ox=landmarksX, oy=landmarksY, xyreso=xyreso, yawreso=yawreso,
            posx=position[0], posy=position[1])
        return self.detected_objects

    def draw_raycasting(self, ax=plt):
        draw_heatmap(self.pmap, self.minx, self.maxx, self.miny, self.maxy, self.xyreso, ax=ax)

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
        return (self.path.x[1], self.path.y[1])

    def get_direction(self):
        """Returns current robot direction"""
        path = self.path
        return -90 + np.rad2deg(np.arctan2(path.y[1] - path.y[0], path.x[1] - path.x[0]))

def random_on_map(count=None):
    """
    Returns random point within the map limits.
    Can return multiple points as one array with count parameter."""
    if count is None:
        count = 1
    return np.random.random(size=(count, 2)) * MAP_SIZE


def main():
    print(__file__ + " start!!")


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
        robots.append(Robot(init_pos=initial_positions[_i], color=colors[_i]))

    time = 0.0

    # RFID positions [[x, y], [x, y]]
    RFID = random_on_map(100)
    # Initialize detected objects with dummy, nonexistent one
    detected_objects = np.array([[100, 100]])
    mapped_objects = np.empty((0,2))
    # Raycasting
    xyreso = 0.25  # x-y grid resolution [m]
    yawreso = np.deg2rad(2.0)  # yaw angle resolution [rad]

    area = 20.0  # animation area length [m]

    # Prepare plots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Raycasting objects coordinates
    landmarksX = RFID[:,0]
    landmarksY = RFID[:,1]

    while SIM_TIME >= time:
        # Frenet Optimal Trajectory
        for robot in robots:
            try:
                robot.plan_path(mapped_objects)
            except AttributeError:
                robot.generate_target_course()
            detected_objects = robot.raycasting(landmarksX, landmarksY, xyreso=xyreso, yawreso=yawreso)
            if not len(detected_objects) == 0:
                mapped_objects = np.unique(np.append(mapped_objects, detected_objects, axis=0), axis=0)
            robot.check_arrived()
            #if robot.check_arrived():
            #    break

        time += DT

        if show_animation:  # pragma: no cover
            ax1.cla()
            ax2.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            # plot frenet
            ax1.set_title(f"Discorvered objects: {len(mapped_objects)}/{100}")
            ax1.set_aspect("equal")
            ax1.grid(True)
            # Plot stuff
            ax1.plot(RFID[:, 0], RFID[:, 1], "xk")
            for robot in robots:
                ax1.plot(robot.get_tx(), robot.get_ty())
                path = robot.get_path()
                ax1.plot(path.x[1:], path.y[1:], "-or")
                ax1.plot(*robot.get_position(), marker=(3, 0, robot.get_direction()), color=robot.get_color())
                robot.draw_raycasting(ax=ax1)

            # Move map with followed robot
            path = robots[FOLLOW_ROBOT].get_path()
            ax1.set_xlim(path.x[1] - area, path.x[1] + area)
            ax1.set_ylim(path.y[1] - area, path.y[1] + area)

            # Draw raycasting
            ax1.plot(landmarksX, landmarksY, "xr")

            # Map plot
            ax2.set_title("Map")
            ax2.set_aspect("equal")
            ax2.grid(True)
            # Draw robots and waypoints
            for robot in robots:
                ax2.plot(*robot.get_position(), marker=(3, 0, robot.get_direction()), color=robot.get_color())
                ax2.plot(*robot.get_waypoints(), "-or")
            # Draw mapped objects
            ax2.plot(mapped_objects[:, 0], mapped_objects[:, 1], "xk")
            ax2.set_xlim(0, MAP_SIZE[0])
            ax2.set_ylim(0, MAP_SIZE[1])
            # Small pause
            plt.pause(0.001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

if __name__ == "__main__":
    main()
