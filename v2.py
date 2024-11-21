from enum import Enum
import carla
import numpy as np
import networkx as nx
from shapely import Polygon
import lib.frenet_optimal_trajectory_planner.FrenetOptimalTrajectory.fot_wrapper as fot
from v2_controller import VehiclePIDController

def kmph_to_mps(speed): return speed/3.6
def mps_to_kmph(speed): return speed*3.6
FOT_HYPERPARAMETERS = {
    "max_speed": kmph_to_mps(15.0),
    "max_accel": 15.0,
    "max_curvature": 15.0,
    "max_road_width_l": 3.0,
    "max_road_width_r": 3.0,
    "d_road_w": 0.5,
    "dt": 0.2,
    "maxt": 10.0,
    "mint": 5.0,
    "d_t_s": 0.5,
    "n_s_sample": 1.0,
    "obstacle_clearance": 0.2,
    "kd": 1.0,
    "kv": 0.1,
    "ka": 0.1,
    "kj": 0.1,
    "kt": 0.1,
    "ko": 0.1,
    "klat": 1.0,
    "klon": 1.0,
    "num_threads": 4,
}
DESTINATION_THRESHOLD = 0.5
WAYPOINT_THRESHOLD = 1.0
REPLAN_THRESHOLD = 5
MAX_SPEED = 15
MIN_SPEED = 1
SLOWDOWN_CONSTANT = 10
NUM_GUIDANCE_WPS = 6
STOP_CONTROL = carla.VehicleControl(brake=1.0)

class Mode(Enum):
    PARKING = 0
    PARKED = 1
    FAILED = 2

# TODO: get data from actual GNSS sensor instead of getting
# perfect vehicle data from CARLA
class CarlaGnssSensor():
    def __init__(self, actor):
        self.actor = actor

    def get_location(self):
        return self.actor.get_location()

    def get_velocity(self):
        return self.actor.get_velocity()

class CarlaCar():
    def __init__(self, world, blueprint, spawn_point, destination, debug=False):
        self.world = world
        self.actor = world.spawn_actor(blueprint, spawn_point)
        self.gnss_sensor = CarlaGnssSensor(self.actor)
        self.car = Car(self.actor.get_location(), self.actor.get_velocity(), destination, self.gnss_sensor)

        self.debug = debug
        if debug:
            self.debug_init(spawn_point, self.car.destination)

    def run_step(self):
        self.actor.apply_control(self.car.run_step())
        if self.debug:
            self.debug_step()

    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

    def debug_step(self):
        self.world.debug.draw_string(self.car.pos, 'X', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)
        for loc in self.car.trajectory:
            self.world.debug.draw_string(loc, 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)
        for pt in self.car.guidance_wps:
            self.world.debug.draw_string(carla.Location(x=pt[0], y=pt[1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=255, b=0), life_time=1.0, persistent_lines=True)

    def destroy(self):
        self.actor.destroy()

    def iou(self):
        actor = self.actor
        car_transform = actor.get_transform()
        car_loc = car_transform.location
        car_angle = car_transform.rotation.yaw
        car_angle = np.deg2rad(actor.get_transform().rotation.yaw)
        car_rotation = np.array([
            [np.cos(car_angle), -np.sin(car_angle)],
            [np.sin(car_angle), np.cos(car_angle)]
        ])
        car_bb = [
            -actor.bounding_box.extent.x, -actor.bounding_box.extent.y,
            actor.bounding_box.extent.x, actor.bounding_box.extent.y
        ]
        car_vertices = [
            np.dot(car_rotation, np.array([car_bb[0], car_bb[1]])) + np.array([car_loc.x, car_loc.y]),
            np.dot(car_rotation, np.array([car_bb[0], car_bb[3]])) + np.array([car_loc.x, car_loc.y]),
            np.dot(car_rotation, np.array([car_bb[2], car_bb[3]])) + np.array([car_loc.x, car_loc.y]),
            np.dot(car_rotation, np.array([car_bb[2], car_bb[1]])) + np.array([car_loc.x, car_loc.y])
        ]
        destination_bb = self.car.destination_bb
        destination_vertices = [(destination_bb[0], destination_bb[1]), (destination_bb[0], destination_bb[3]), (destination_bb[2], destination_bb[3]), (destination_bb[2], destination_bb[1])]

        # Debug bounding boxes
        # self.world.debug.draw_string(carla.Location(x=car_vertices[0][0], y=car_vertices[0][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=car_vertices[1][0], y=car_vertices[1][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=car_vertices[2][0], y=car_vertices[2][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=car_vertices[3][0], y=car_vertices[3][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)

        # self.world.debug.draw_string(carla.Location(x=destination_vertices[0][0], y=destination_vertices[0][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=destination_vertices[1][0], y=destination_vertices[1][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=destination_vertices[2][0], y=destination_vertices[2][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=destination_vertices[3][0], y=destination_vertices[3][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)

        car_polygon = Polygon(car_vertices)
        destination_polygon = Polygon(destination_vertices)
        iou = car_polygon.intersection(destination_polygon).area / car_polygon.union(destination_polygon).area
        return iou

class Car():
    def __init__(self, pos, vel, destination, gnss_sensor):
        self.pos = pos
        self.vel = vel
        self.ps = 0
        self.obs = []
        self.lane_wps = []
        self.guidance_wps = []
        self.destination = carla.Location(x=(destination[0] + destination[2]) / 2, y=(destination[1] + destination[3]) / 2)
        self.destination_bb = destination
        self.fast_controller = VehiclePIDController({'K_P': 2, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})
        self.slow_controller = VehiclePIDController({'K_P': 8, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})
        self.gnss_sensor = gnss_sensor
        self.trajectory = []
        self.mode = Mode.PARKING

    def perceive(self):
        self.pos = self.gnss_sensor.get_location()
        self.vel = self.gnss_sensor.get_velocity()
        # TODO: get/update obstacles and lane waypoints from sensor data only,
        # adding data to graph as needed

    def plan(self):
        # TODO: handle planning for exploration phase
        # if we're at destination, stop
        # TODO: also stop if unexpected obstacle detected
        pos = self.pos
        destination = self.destination
        distance_to_destination = pos.distance(destination)
        if self.mode == Mode.PARKED or distance_to_destination < DESTINATION_THRESHOLD:
            self.mode = Mode.PARKED
            return STOP_CONTROL

        # plan guidance wps if needed
        guidance_wps = self.guidance_wps
        lane_wps = self.lane_wps
        if not guidance_wps:
            # find closest waypoints to pos and destination
            pos_lane_wp = min(range(len(lane_wps)), key=lambda i: pos.distance(carla.Location(x=lane_wps[i][0], y=lane_wps[i][1])))
            destination_lane_wp = min(range(len(lane_wps)), key=lambda i: destination.distance(carla.Location(x=lane_wps[i][0], y=lane_wps[i][1])))

            # find shortest path
            G = nx.Graph()
            for i, waypoint in enumerate(lane_wps):
                G.add_node(i, pos=(waypoint[0], waypoint[1]))
            for i in range(len(lane_wps) - 1):
                G.add_edge(i, i + 1)
            guidance_wps.append([pos.x, pos.y])
            for wp in map(lambda i: lane_wps[i], nx.shortest_path(G, source=pos_lane_wp, target=destination_lane_wp)):
                guidance_wps.append(wp)

            # add additional guidance waypoints from parking spot
            # TODO: handle case where longer side is along different axis
            destination_bb = self.destination_bb
            start_x, end_x = destination_bb[0] - 2, destination_bb[2]
            if abs(start_x - guidance_wps[-1][0]) > abs(end_x - guidance_wps[-1][0]):
                start_x, end_x = end_x + 2, start_x + 2
            for wp_x, wp_y in zip(
                np.linspace(start_x, end_x, NUM_GUIDANCE_WPS).tolist(),
                [(destination_bb[1] + destination_bb[3])/2] * NUM_GUIDANCE_WPS,
            ):
                guidance_wps.append([wp_x, wp_y])

        # remove visited points from trajectory
        trajectory = self.trajectory
        num_to_remove = 0
        for loc in trajectory:
            if pos.distance(loc) < WAYPOINT_THRESHOLD:
                num_to_remove += 1
            else:
                break
        if num_to_remove > 0:
            trajectory = self.trajectory = trajectory[num_to_remove:]

        # replan trajectory if needed
        target_speed = (MAX_SPEED-MIN_SPEED) * distance_to_destination / (distance_to_destination + SLOWDOWN_CONSTANT) + MIN_SPEED
        if len(trajectory) < REPLAN_THRESHOLD:
            ps = self.ps
            vel = self.vel
            obs = self.obs

            # use FOT planner
            initial_conditions = {
                'ps': ps,
                'target_speed': kmph_to_mps(target_speed),
                'pos': np.array([pos.x, pos.y]),
                'vel': np.array([kmph_to_mps(vel.x), kmph_to_mps(vel.y)]),
                'wp': np.array(guidance_wps),
                'obs': np.array(obs)
            }
            result_x, result_y, speeds, ix, iy, iyaw, d, s, speeds_x, speeds_y, \
                misc, costs, success = fot.run_fot(initial_conditions, FOT_HYPERPARAMETERS)
            ps = self.ps = misc['s']

            # truncate points that are too close
            new_trajectory = []
            for x, y in zip(result_x, result_y):
                new_loc = carla.Location(x=x, y=y, z=pos.z)
                if pos.distance(new_loc) > WAYPOINT_THRESHOLD:
                    new_trajectory.append(new_loc)
            if success and new_trajectory:
                trajectory = self.trajectory = new_trajectory

        if not trajectory: self.mode = Mode.FAILED; return STOP_CONTROL
        ctrl = (self.slow_controller if mps_to_kmph(self.vel.length()) < SLOWDOWN_CONSTANT else self.fast_controller).run_step(self.vel, target_speed, pos, trajectory[0])
        return ctrl

    def run_step(self):
        self.perceive()
        return self.plan()