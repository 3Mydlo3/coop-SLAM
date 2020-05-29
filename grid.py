import numpy as np

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
        """Returns slice of the map corresponding to grid coordinates"""
        self.shift = self.grid_coordinates * self.grid_size
        grid = self.map.map_coverage[self.shift[0] : self.shift[0] + self.grid_size, self.shift[1] : self.shift[1] + self.grid_size]
        return grid

    def is_covered(self):
        """Checks if grid is fully covered"""
        return np.all(self.grid > 0.01)

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
