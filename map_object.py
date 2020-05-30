import numpy as np

class MapObject:
    def __init__(self, init_pos, id_, map_):
        """"""
        self.position = init_pos
        if np.shape(self.position) == (1, 2):
            self.position = self.position[0]
        self.id = id_
        self.map = map_
        self.information = 0.0
        self.discovered = False
        self.relocated_object = None

    def __eq__(self, other):
        """For == comparison support"""
        if isinstance(other, MapObject):
            return True if self.id == other.id else False
        else:
            return NotImplemented

    def __hash__(self):
        return hash(str(self.position))

    def get_id(self):
        return self.id

    def get_position(self):
        return self.position

    def get_information(self):
        return self.information

    def has_relocated(self):
        return True if self.relocated_object is not None else False

    def is_visible(self):
        """Checks whether objects any robot can see the object"""
        try:
            self.information = self.map.get_position_coverage(*self.position) * 2
        except IndexError:
            print("DUPA")
        return self.information >= 0.99

    def discover(self):
        self.discovered = True

    def relocated(self, new_object):
        """Updates reference object where relocated"""
        self.relocated_object = new_object

    def was_discovered(self):
        return self.discovered
