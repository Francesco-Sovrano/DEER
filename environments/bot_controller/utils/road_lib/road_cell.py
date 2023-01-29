import numpy as np
from .road_agent import RoadAgent

class RoadCell(RoadAgent):
    def __init__(self, i=-1, j=-1):
        self.current_position = (i, j)
        super().__init__()

    def culture_properties(self):
        if self.road_culture is None:
            return None
        return self.road_culture.__dict__.get("properties", None)

    def build_features(self):
        self.features_tuple = tuple(
            0 if not self[prop] else 1
            for prop in self.sorted_properties
        )
        self.features = np.array(self.features_tuple, dtype=np.int8)
