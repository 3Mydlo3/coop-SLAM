import numpy as np
import matplotlib.pyplot as plt

from grid import Grid
from params import MAP_SIZE, GRID_SIZE, OBJECTS_COUNT, DECAY_RATE

class Map:
    def __init__(self):
        self.size = np.array(MAP_SIZE)
        self.objects = self._generate_objects(OBJECTS_COUNT)
        self.mapped_objects = np.empty((0,2))
        self.robots = []
        self.map_coverage = self._prepare_map_coverage()
        self.grids = self._prepare_grids(GRID_SIZE)

    def _generate_objects(self, objects_count):
        """Generates map objects"""
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
        """Returns all free, nonexplored grids"""
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
        return np.count_nonzero(self.map_coverage > 0.01)/self.map_coverage_size * 100

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

    def map_decay(self, rate=DECAY_RATE):
        """Handles coverage data decay on map"""
        self.map_coverage[self.map_coverage > 0.01] -= rate

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
