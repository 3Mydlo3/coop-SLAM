"""

Ray casting 2D grid map example

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np
import matplotlib.pyplot as plt

EXTEND_AREA = 30.0
RAYS_RANGE = 14.0 # maximum rays range [m]

show_animation = True


def calc_grid_map_config(ox, oy, xyreso, posx=0, posy=0):
    minx = round(posx - EXTEND_AREA / 2.0)
    miny = round(posy - EXTEND_AREA / 2.0)
    maxx = round(posx + EXTEND_AREA / 2.0)
    maxy = round(posy + EXTEND_AREA / 2.0)
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    return minx, miny, maxx, maxy, xw, yw


class precastDB:

    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.d = 0.0
        self.angle = 0.0
        self.ix = 0
        self.iy = 0

    def __str__(self):
        return str(self.px) + "," + str(self.py) + "," + str(self.d) + "," + str(self.angle)


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0

    return angle


def precasting(minx, miny, xw, yw, xyreso, yawreso, posx=0, posy=0):

    precast = [[] for i in range(int(round((math.pi * 2.0) / yawreso)) + 1)]

    for ix in range(xw):
        for iy in range(yw):
            px = ix * xyreso + minx
            py = iy * xyreso + miny

            d = math.hypot(px - posx, py - posy)
            angle = atan_zero_to_twopi(py - posy, px - posx)
            angleid = int(math.floor(angle / yawreso))

            pc = precastDB()

            pc.px = px
            pc.py = py
            pc.d = d
            pc.ix = ix
            pc.iy = iy
            pc.angle = angle

            precast[angleid].append(pc)

    return precast


def generate_ray_casting_grid_map(objects, xyreso, yawreso, posx=0, posy=0):

    objects_positions = np.empty(shape=(len(objects), 2))
    for i in range(0, len(objects)):
        objects_positions[i, :] = objects[i].get_position()
    ox = objects_positions[:, 0]
    oy = objects_positions[:, 1]

    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso, posx=posx, posy=posy)

    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    precast = precasting(minx, miny, xw, yw, xyreso, yawreso, posx=posx, posy=posy)

    obstacles_in_range = []
    for i in range(0, len(objects)):
        x, y = objects_positions[i, :]
        # Check if given obstacle is in rays range
        d = math.hypot(x - posx, y - posy)
        if d <= RAYS_RANGE:
            obstacles_in_range.append(objects[i])

            angle = atan_zero_to_twopi(y - posy, x - posx)
            angleid = int(math.floor(angle / yawreso))

            gridlist = precast[angleid]

            ix = int(round((x - minx) / xyreso))
            iy = int(round((y - miny) / xyreso))

            for grid in gridlist:
                if grid.d > d:
                    pmap[grid.ix][grid.iy] = 0.5

            pmap[ix][iy] = 1.0

    return pmap, minx, maxx, miny, maxy, obstacles_in_range


def get_heatmap(data, minx, maxx, miny, maxy, xyreso):
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso),
                    slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
    # Slice one sample for compatibility with data size
    x, y = x[:-1, :-1] , y[:-1, :-1]
    # Create circle showing rays range
    data = np.where(np.logical_and(np.sqrt(np.power(x - np.average([minx, maxx]), 2) + np.power(y - np.average([miny, maxy]), 2)) < RAYS_RANGE, np.array(data) == 0), 0.01, data)
    # Filter all data outside rays range
    data = np.where(np.sqrt(np.power(x - np.average([minx, maxx]), 2) + np.power(y - np.average([miny, maxy]), 2)) > RAYS_RANGE, 0, data)
    return x, y, data
