import numpy as np

class RoadAgent:
    def __init__(self):
        self.road_culture = None

    def __getitem__(self, item):
        return self.__dict__.get(item, None)

    # def __setitem__(self, key, value):
    #     self.__dict__[key] = value

    def culture_properties(self):
        if self.road_culture is None:
            return None
        return self.road_culture.__dict__.get("agent_properties", None)

    def binary_features(self, as_tuple=False): # O(1)
        return self.features if not as_tuple else self.features_tuple

    def set_culture(self, culture):
        self.road_culture = culture
        if self.culture_properties() is None:
            print("RoadCell::set_culture: Culture {} has no properties.".format(culture.name))
            return
        for p, v in self.culture_properties().items():
            self.__setattr__(p, v)
        self.sorted_properties = sorted(self.culture_properties().keys())
        ####
        self.build_features()

    def build_features(self):
        self.features_tuple = tuple(
            0 if not self[prop] else 1
            for prop in self.sorted_properties
            if prop != "Speed"
        )
        self.features = np.array(self.features_tuple, dtype=np.int8)

    def assign_property_value(self, property_, value):
        # if hasattr(self, property_) is False:
        #     print("RoadCell::assign_property_value: Property {} not found within road cell.".format(property_))
        #     return
        self.__setattr__(property_, value)
        self.build_features()
