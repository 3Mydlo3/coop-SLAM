from raycasting_grid_map import *
from frenet_optimal_trajectory import *

SIM_TIME = 50.0 # simulation time [s]

def main():
    print(__file__ + " start!!")

    initial_position = [0, 0]

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [-5.0, 20.0]])

    # Initialize detected objects with dummy, nonexistent one
    detected_objects = np.array([[100, 100]])

    # Raycasting
    xyreso = 0.25  # x-y grid resolution [m]
    yawreso = np.deg2rad(2.0)  # yaw angle resolution [rad]

    # Frenet Trajectory
    # way points
    wx = [initial_position[0], 20, -5]
    wy = [initial_position[1], 7, 18]

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    # initial state
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_d = initial_position[0]  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = initial_position[1]  # current course position

    area = 20.0  # animation area length [m]

    while SIM_TIME >= time:
        # Frenet Optimal Trajectory
        path = frenet_optimal_planning(
        csp, s0, c_speed, c_d, c_d_d, c_d_dd, detected_objects)

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

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            # plot frenet
            plt.plot(tx, ty)
            plt.plot(RFID[:, 0], RFID[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])

            plt.axis("equal")
            plt.grid(True)
            # Draw raycasting
            draw_heatmap(pmap, minx, maxx, miny, maxy, xyreso)
            plt.plot(landmarksX, landmarksY, "xr")
            # Small pause
            plt.pause(0.001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

if __name__ == "__main__":
    main()
