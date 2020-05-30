import numpy as np
import matplotlib.pyplot as plt

from robot import Robot
from map import Map
from params import ROBOTS_COUNT, DT, SHOW_ANIMATION, ANIM_AREA, FOLLOW_ROBOT, MAP_SIZE

def main():
    print(__file__ + " start!!")

    # Create map and get objects
    map_entity = Map()

    # Create robots
    robots = []
    for _i in range(0, ROBOTS_COUNT):
        robots.append(Robot(map_=map_entity, id_=_i))

    time = 0.0

    # Prepare plots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    while map_entity.get_current_coverage() < 100:
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
            ax1.cla()
            ax2.cla()
            # Obtain all mapped objects
            mapped_objects = map_entity.get_mapped_objects_positions()
            relocated_objects = map_entity.get_relocated_objects_positions()
            # For stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            # Plot robot surroundings
            ax1.set_title(f"Discovered: {map_entity.get_current_coverage():.2f} %")
            ax1.set_aspect("equal")
            ax1.grid(True)
            # Plot objects
            objects = map_entity.get_all_objects_positions()
            ax1.plot(objects[:, 0], objects[:, 1], "xk")
            # Plot all robots paths and positions
            for robot in robots:
                try:
                    ax1.plot(robot.get_tx(), robot.get_ty())
                    path = robot.get_path()
                    ax1.plot(path.x[1:], path.y[1:], "-or")
                except AttributeError:
                    # Path is not available
                    pass
                ax1.plot(*robot.get_position(), marker=(3, 0, robot.get_direction()), color=robot.get_color())
            # Draw scanning area
            map_entity.draw_current_scanning_area(ax=ax1)
            # Move map with followed robot
            robot_position = robots[FOLLOW_ROBOT].get_position()
            ax1.set_xlim(robot_position[0] - ANIM_AREA, robot_position[0] + ANIM_AREA)
            ax1.set_ylim(robot_position[1] - ANIM_AREA, robot_position[1] + ANIM_AREA)
            # Plot mapped objects
            ax1.plot(mapped_objects[:, 0], mapped_objects[:, 1], "xr")
            ax1.plot(relocated_objects[:, 0], relocated_objects[:, 1], "xy")

            # Plot map
            ax2.set_title("Map")
            ax2.set_aspect("equal")
            ax2.grid(True)
            ax2.set_xlim(0, MAP_SIZE[0])
            ax2.set_ylim(0, MAP_SIZE[1])
            # Draw robots and waypoints
            for robot in robots:
                ax2.plot(*robot.get_position(), marker=(3, 0, robot.get_direction()), color=robot.get_color())
                if robot.get_assigned_grid() is not None:
                    ax2.plot(*robot.get_waypoints(), "-or")
            # Draw mapped objects
            ax2.plot(mapped_objects[:, 0], mapped_objects[:, 1], "xr")
            # Draw map coverage info
            map_entity.draw_map_coverage(ax=ax2)

            # Small pause
            plt.pause(0.001)

    print("Finish")
    if SHOW_ANIMATION:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

if __name__ == "__main__":
    main()
