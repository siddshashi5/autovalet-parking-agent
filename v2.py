import carla
import numpy as np
import lib.frenet_optimal_trajectory_planner.FrenetOptimalTrajectory.fot_wrapper as fot
from agent.controller import VehiclePIDController

HOST = '127.0.0.1'
PORT = 2000
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

class Car():
    def __init__(self, world, blueprint, spawn_point, destination):
        self.actor = world.spawn_actor(blueprint, spawn_point)
        self.pos = self.actor.get_location()
        self.vel = self.actor.get_velocity()
        self.ps = 0
        self.obs = []
        self.destination = destination
        self.controller = VehiclePIDController(self.actor, {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.05})
        # TODO: mount sensors

    def perceive(self):
        self.pos = self.actor.get_location()
        self.vel = self.actor.get_velocity()
        # TODO: get/update obstacles from sensor data only

    def plan(self):
        # TODO: only replan trajectory if necessary
        ps = self.ps
        pos = self.pos
        destination = self.destination
        obs = self.obs
        initial_conditions = {
            'ps': ps,
            'target_speed': 20,
            'pos': np.array([pos.x, pos.y]),
            'vel': np.array([0, 0]),
            'wp': np.array([[pos.x, pos.y], [destination.x, destination.y]]),
            'obs': np.array(obs)
        }
        result_x, result_y, speeds, ix, iy, iyaw, d, s, speeds_x, speeds_y \
            misc, costs, success = fot.run_fot(initial_conditions, FOT_HYPERPARAMETERS)

        trajectory = []
        for x, y in zip(result_x, result_y):
            trajectory.append(carla.Location(x=x, y=y, z=pos.z))

        # TODO: stop if reached destination OR if unexpected obstacle detected by sensor
        self.actor.apply_control(self.controller.run_step(5, trajectory[1]))

    def run_step(self):
        self.perceive()
        self.plan()

    def destroy():
        self.actor.destroy()

def init_simulation():
    # connect to client
    client = carla.Client(HOST, PORT)
    client.set_timeout(2.0)

    # setup world map
    world = client.load_world('Town04_Opt')
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    # TODO: setup spectator

    return world

def main():
    print(f"starting simulation on {HOST}:{PORT}")
    world = init_simulation()
    blueprint = world.get_blueprint_library().filter('vehicle.mercedes.coupe_2020')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    car = Car(world, blueprint, spawn_point)

    try:
        while True:
            world.wait_for_tick()
            car.run_step()
    except KeyboardInterrupt:
        print("stopping simulation")
    finally:
        car.destroy()

if __name__ == '__main__':
    main()
