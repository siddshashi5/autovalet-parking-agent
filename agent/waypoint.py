import carla

class Waypoint():
    def __init__(self, x, y):
        self.transform = carla.Transform()
        self.transform.location = carla.Location(x=x, y=y)
        self.transform.rotation = carla.Rotation()