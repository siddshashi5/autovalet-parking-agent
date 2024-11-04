import carla
import numpy as np
import lib.frenet_optimal_trajectory_planner.FrenetOptimalTrajectory.fot_wrapper as fot
from v2_controller import VehiclePIDController

FOT_HYPERPARAMETERS = {
    "max_speed": 15.0,
    "max_accel": 15.0,
    "max_curvature": 15.0,
    "max_road_width_l": 3.0,
    "max_road_width_r": 3.0,
    "d_road_w": 0.5,
    "dt": 0.2,
    "maxt": 5.0,
    "mint": 2.0,
    "d_t_s": 0.5,
    "n_s_sample": 2.0,
    "obstacle_clearance": 0.6,
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
        self.debug_last_trajectory = None
        if debug:
            self.debug_init(spawn_point, self.car.destination)

    def run_step(self):
        self.actor.apply_control(self.car.run_step())
        if self.debug:
            self.debug_step()

    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

        # TEMP: debug guidance waypoints
        guidance_wps = self.car.guidance_wps
        for pt in guidance_wps:
            self.world.debug.draw_string(carla.Location(x=pt[0], y=pt[1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=255, b=0), life_time=120.0, persistent_lines=True)

    def debug_step(self):
        trajectory = self.car.trajectory
        if trajectory == self.debug_last_trajectory: return
        self.debug_last_trajectory = trajectory
        for loc in trajectory:
            self.world.debug.draw_string(loc, 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)

    def destroy(self):
        self.actor.destroy()

class Car():
    def __init__(self, pos, vel, destination, gnss_sensor):
        self.pos = pos
        self.vel = vel
        self.ps = 0
        self.obs = []
        # TODO: handle case where longer side is along different axis
        self.guidance_wps = [[wp_x, wp_y] for wp_x, wp_y in zip(
            np.linspace(destination[0], destination[2], NUM_GUIDANCE_WPS).tolist(),
            [(destination[1] + destination[3])/2] * NUM_GUIDANCE_WPS,
        )]
        self.destination = carla.Location(x=(destination[0] + destination[2]) / 2, y=(destination[1] + destination[3]) / 2)
        self.controller = VehiclePIDController({'K_P': 8, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})
        self.gnss_sensor = gnss_sensor
        self.trajectory = []
        # TODO: mount sensors

    def perceive(self):
        self.pos = self.gnss_sensor.get_location()
        self.vel = self.gnss_sensor.get_velocity()
        # TODO: get/update obstacles from sensor data only

    def plan(self):
        # TODO: handle planning for exploration phase
        # if we're at destination, stop
        # TODO: also stop if unexpected obstacle detected
        pos = self.pos
        destination = self.destination
        distance_to_destination = pos.distance(destination)
        if distance_to_destination < DESTINATION_THRESHOLD:
            return STOP_CONTROL

        # remove visited points from trajectory
        trajectory = self.trajectory
        num_to_remove = 0

        for loc in trajectory:
            if pos.distance(loc) < WAYPOINT_THRESHOLD:
                num_to_remove += 1
            else:
                break

        trajectory = self.trajectory = trajectory[num_to_remove:]

        # replan trajectory if needed
        target_speed = (MAX_SPEED-MIN_SPEED) * distance_to_destination / (distance_to_destination + SLOWDOWN_CONSTANT) + MIN_SPEED
        if len(trajectory) < REPLAN_THRESHOLD:
            ps = self.ps
            vel = self.vel
            obs = self.obs

            # truncate visited guidance waypoints
            num_truncate = 0
            for wp in self.guidance_wps:
                if pos.distance(carla.Location(x=wp[0], y=wp[1])) < WAYPOINT_THRESHOLD:
                    num_truncate += 1
                else:
                    break
            guidance_wps = self.guidance_wps = self.guidance_wps[num_truncate:]

            # use FOT planner
            initial_conditions = {
                'ps': ps,
                'target_speed': target_speed,
                'pos': np.array([pos.x, pos.y]),
                'vel': np.array([vel.x, vel.y]),
                'wp': np.array([[pos.x, pos.y]] + guidance_wps),
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
            if new_trajectory:
                trajectory = self.trajectory = new_trajectory

        if not trajectory: return STOP_CONTROL
        ctrl = self.controller.run_step(self.vel, target_speed, pos, trajectory[0])
        return ctrl

    def run_step(self):
        self.perceive()
        return self.plan()