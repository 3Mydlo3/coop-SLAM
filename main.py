from raycasting_grid_map import *
from frenet_optimal_trajectory import *

SIM_TIME = 50.0 # simulation time [s]
MAP_SIZE = [100, 100] # X and Y size of map

def main():
    print(__file__ + " start!!")

    initial_position = [0, 0]

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, 2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [5.0, 20.0]])
    # Initialize detected objects with dummy, nonexistent one
    detected_objects = np.array([[100, 100]])
    mapped_objects = np.empty((0,2))
    # Raycasting
    xyreso = 0.25  # x-y grid resolution [m]
    yawreso = np.deg2rad(2.0)  # yaw angle resolution [rad]

    # Frenet Trajectory
    # way points
    wx = [initial_position[0], 20, 1]
    wy = [initial_position[1], 7, 18]

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    # initial state
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_d = initial_position[0]  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = initial_position[1]  # current course position

    area = 20.0  # animation area length [m]

    # Prepare plots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    while SIM_TIME >= time:
        # Frenet Optimal Trajectory
        path = frenet_optimal_planning(
        csp, s0, c_speed, c_d, c_d_d, c_d_dd, mapped_objects)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]

        # Check if arrived
        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        time += DT

        # Raycasting
        landmarksX = RFID[:,0]
        landmarksY = RFID[:,1]
        pmap, minx, maxx, miny, maxy, xyreso, detected_objects = generate_ray_casting_grid_map(
            ox=landmarksX, oy=landmarksY, xyreso=xyreso, yawreso=yawreso,
            posx=path.x[1], posy=path.y[1])

        if not len(detected_objects) == 0:
            mapped_objects = np.unique(np.append(mapped_objects, detected_objects, axis=0), axis=0)
        robot_direction = -90 + np.rad2deg(np.arctan2(path.y[1] - path.y[0], path.x[1] - path.x[0]))

        if show_animation:  # pragma: no cover
            ax1.cla()
            ax2.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            # plot frenet
            ax1.set_title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            ax1.set_aspect("equal")
            ax1.grid(True)
            # Plot stuff
            ax1.plot(tx, ty)
            ax1.plot(RFID[:, 0], RFID[:, 1], "xk")
            ax1.plot(path.x[1:], path.y[1:], "-or")
            ax1.plot(path.x[1], path.y[1], marker=(3, 0, robot_direction), color='tab:blue')
            ax1.set_xlim(path.x[1] - area, path.x[1] + area)
            ax1.set_ylim(path.y[1] - area, path.y[1] + area)

            # Draw raycasting
            draw_heatmap(pmap, minx, maxx, miny, maxy, xyreso, ax=ax1)
            ax1.plot(landmarksX, landmarksY, "xr")

            # Map plot
            ax2.set_title("Map")
            ax2.set_aspect("equal")
            ax2.grid(True)
            # Draw robot
            ax2.plot(path.x[1], path.y[1], marker=(3, 0, robot_direction), color='tab:blue')
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
