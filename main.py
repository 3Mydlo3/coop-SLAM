import numpy as np
import matplotlib.pyplot as plt

from robot import Robot
from map import Map
from params import ROBOTS_COUNT, DT, SHOW_ANIMATION, ANIM_AREA, FOLLOW_ROBOT, MAP_SIZE

end = False

def key_pressed(event):
    global end
    print("EVENT")
    if event.key == 'escape':
        print("END")
        end = True

def main():
    # Create map and get objects
    map_entity = Map()

    # Create robots
    robots = []
    for _i in range(0, ROBOTS_COUNT):
        robots.append(Robot(map_=map_entity, id_=_i))

    time = 0.0

    # Prepare plots
    fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(8, 8))
    gs = axs[2, 3].get_gridspec()
    # remove the underlying axes
    for i in range(-1, -4, -1):
        for ax in axs[:, i]:
            ax.remove()
    axmap = fig.add_subplot(gs[:, -3:])
    fig.tight_layout()
    plots = fig.get_axes()
    robot_plots = plots[:-1]

    while map_entity.get_current_coverage() < 100 and not end:
        # Objects random relocation
        map_entity.randomly_relocate_objects()
        # Map decay
        map_entity.map_decay()
        # Simulate robots
        for robot in robots:
            robot.simulate()

        time += DT

        if SHOW_ANIMATION:
            # Clear plots
            for plot in plots:
                plot.cla()
            # Obtain all mapped objects
            mapped_objects = map_entity.get_mapped_objects_positions()
            relocated_objects = map_entity.get_relocated_objects_positions()
            # For stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_press_event', key_pressed)

            for i in range(0, 3):
                axrb = robot_plots[i]
                robot = robots[i]
                # Plot robot surroundings
                axrb.set_title(f"Robot {robot.get_id()}")
                axrb.set_aspect("equal")
                axrb.grid(True)
                # Plot objects
                objects = map_entity.get_all_objects_positions()
                axrb.plot(objects[:, 0], objects[:, 1], "xk")
                # Plot robot path and robots positions
                try:
                    axrb.plot(robot.get_tx(), robot.get_ty())
                    path = robot.get_path()
                    axrb.plot(path.x[1:], path.y[1:], "-or")
                except AttributeError:
                    # Path is not available
                    pass
                for _robot in robots:
                    axrb.plot(*_robot.get_position(), marker=(3, 0, _robot.get_direction()), color=_robot.get_color())
                # Draw scanning area
                map_entity.draw_current_scanning_area(ax=axrb)
                # Move map with followed robot
                robot_position = robot.get_position()
                axrb.set_xlim(robot_position[0] - ANIM_AREA, robot_position[0] + ANIM_AREA)
                axrb.set_ylim(robot_position[1] - ANIM_AREA, robot_position[1] + ANIM_AREA)
                # Plot mapped objects
                axrb.plot(mapped_objects[:, 0], mapped_objects[:, 1], "xr")
                axrb.plot(relocated_objects[:, 0], relocated_objects[:, 1], "xy")

            # Plot map
            axmap.set_title(f"Map, discovered: {map_entity.get_current_coverage():.2f} %")
            axmap.set_aspect("equal")
            axmap.grid(True)
            axmap.set_xlim(0, MAP_SIZE[0])
            axmap.set_ylim(0, MAP_SIZE[1])
            # Draw robots and waypoints
            for robot in robots:
                axmap.plot(*robot.get_position(), marker=(3, 0, robot.get_direction()), color=robot.get_color())
                if robot.get_assigned_grid() is not None:
                    axmap.plot(*robot.get_waypoints(), "-ob")
            # Draw mapped objects
            axmap.plot(mapped_objects[:, 0], mapped_objects[:, 1], "xr")
            # Draw map coverage info
            map_entity.draw_map_coverage(ax=axmap)

            # Small pause
            plt.pause(0.001)

    print("Finish")
    if SHOW_ANIMATION:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

if __name__ == "__main__":
    main()
