import carla
import numpy as np
import lib.frenet_optimal_trajectory_planner.FrenetOptimalTrajectory.fot_wrapper as fot
from v2_controller import VehiclePIDController

FOT_HYPERPARAMETERS = {
    "max_speed": 25.0,
    "max_accel": 15.0,
    "max_curvature": 15.0,
    "max_road_width_l": 5.0,
    "max_road_width_r": 5.0,
    "d_road_w": 0.5,
    "dt": 0.2,
    "maxt": 5.0,
    "mint": 2.0,
    "d_t_s": 0.5,
    "n_s_sample": 2.0,
    "obstacle_clearance": 0.1,
    "kd": 1.0,
    "kv": 0.1,
    "ka": 0.1,
    "kj": 0.1,
    "kt": 0.1,
    "ko": 0.1,
    "klat": 1.0,
    "klon": 1.0,
    "num_threads": 0,
}
REPLAN_THRESHOLD = 5
TARGET_SPEED = 5
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
            self.debug_init(spawn_point, destination)

    def run_step(self):
        # spectator_rotation = self.actor.get_transform().rotation
        # spectator_rotation.pitch -= 20
        # spectator_transform = carla.Transform(self.actor.get_transform().transform(carla.Location(x=-10,z=5)), spectator_rotation)
        # self.world.get_spectator().set_transform(spectator_transform)

        self.actor.apply_control(self.car.run_step())
        if self.debug:
            self.debug_step()

    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

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
        self.destination = destination
        self.controller = VehiclePIDController({'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.05})
        self.gnss_sensor = gnss_sensor
        self.trajectory = []
        # TODO: mount sensors

    def perceive(self):
        self.pos = self.gnss_sensor.get_location()
        self.vel = self.gnss_sensor.get_velocity()
        # TODO: get/update obstacles from sensor data only

    def plan(self):
        # if we're at destination, stop
        # TODO: also stop if unexpected obstacle detected
        pos = self.pos
        destination = self.destination
        if pos.distance(destination) < 1.0:
            return STOP_CONTROL

        # remove visited points from trajectory
        trajectory = self.trajectory
        num_to_remove = 0

        for loc in trajectory:
            if pos.distance(loc) < 1.0:
                num_to_remove += 1
            else:
                break

        trajectory = self.trajectory = trajectory[num_to_remove:]

        # replan trajectory if needed
        if len(trajectory) < REPLAN_THRESHOLD:
            ps = self.ps
            obs = self.obs
            initial_conditions = {
                'ps': ps,
                'target_speed': TARGET_SPEED,
                'pos': np.array([pos.x, pos.y]),
                'vel': np.array([0, 0]),
                'wp': np.array([[pos.x, pos.y], [destination.x, destination.y]]),
                'obs': np.array(obs)
            }
            result_x, result_y, speeds, ix, iy, iyaw, d, s, speeds_x, speeds_y, \
                misc, costs, success = fot.run_fot(initial_conditions, FOT_HYPERPARAMETERS)

            trajectory = []
            for x, y in zip(result_x, result_y):
                trajectory.append(carla.Location(x=x, y=y, z=pos.z))
            self.trajectory = trajectory

        if not trajectory: return STOP_CONTROL
        return self.controller.run_step(self.vel, TARGET_SPEED, self.pos, trajectory[1])

    def run_step(self):
        self.perceive()
        return self.plan()