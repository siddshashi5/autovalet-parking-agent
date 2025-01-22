from v2_experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spectator_follow,
    town04_get_drivable_graph,
    obstacle_map_from_bbs,
    mask_obstacle_map,
    DELTA_SECONDS
)

# For lane waypoint hack
from parking_position import (
    parking_lane_waypoints_Town04
)

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

SCENARIOS = [
    # (17, [16, 18]),
    # (18, [17, 19]),
    # (19, [18, 20]),
    # (20, [19, 21]),
    (21, [20, 22]),
    (22, [21, 23]),
    # (23, [22, 24]),
    # (24, [23, 25]),
    # (25, [24, 26]),
    # (26, [25, 27]),
    # (27, [26, 28]),
    # (28, [27, 29]),
    # (29, [28, 30]),
    # (30, [29, 31]),
    # (31, [30, 32]),
    # (32, [31, 33]),
    # (33, [32, 34]),
    # (34, [33, 35]),
    # (35, [34, 36]),
    # (36, [35, 37]),
    # (37, [36, 38]),
    # (38, [37, 39]),
    # (39, [38, 40]),
    # (40, [39, 41]),
    # (41, [40, 42]),
    # (42, [41, 43]),
    # (43, [42, 44]),
    # (44, [43, 45]),
    # (45, [44, 46]),
    # (46, [45, 47]),
    # (47, [46, 48]),
]
NUM_RANDOM_CARS = 25
PERCEPTION_LATENCY = 15

def run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file):
    try:
        # load parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(world, parked_spots, destination_parking_spot, NUM_RANDOM_CARS)

        # spawn traffic cone with blueprint static.prop.trafficcone01
        import carla
        traffic_cone_bp = world.get_blueprint_library().find('static.prop.trafficcone01')
        traffic_cones = []
        traffic_cone_bbs = []
        traffic_cone_locations = [
            carla.Location(x=284, y=-235+5, z=0.3),
            carla.Location(x=287, y=-230+5, z=0.3),
            # carla.Location(x=284, y=-235, z=0.3),
        ]
        for traffic_cone_location in traffic_cone_locations:
            traffic_cone_transform = carla.Transform(traffic_cone_location)
            traffic_cone = world.try_spawn_actor(traffic_cone_bp, traffic_cone_transform)
            traffic_cone.set_simulate_physics(False)
            traffic_cones.append(traffic_cone)
            traffic_cone_bbs.append([
                traffic_cone_location.x - traffic_cone.bounding_box.extent.x, traffic_cone_location.y - traffic_cone.bounding_box.extent.y,
                traffic_cone_location.x + traffic_cone.bounding_box.extent.x, traffic_cone_location.y + traffic_cone.bounding_box.extent.y
            ])

        # spawn walker
        # walker_bp = world.get_blueprint_library().filter('walker.*')
        # walker_location = carla.Location(x=285, y=-230-2, z=0.3)
        # walker_transform = carla.Transform(walker_location)
        # walker = world.try_spawn_actor(walker_bp[0], walker_transform)
        world.tick()

        # load car
        car = town04_spawn_ego_vehicle(world, destination_parking_spot)
        recording_cam = car.init_recording(recording_file)

        # HACK: enable perfect perception of parked cars
        car.car.obs = obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs)
        mask_obstacle_map(car.car.obs, 0, 0)
        # visualize 1 hot map
        plt.cla()
        plt.imshow(car.car.obs.obs, cmap='gray')
        plt.savefig('obs_map.png')

        # HACK: set lane waypoints to guide parking in adjacent lanes
        car.car.lane_wps = parking_lane_waypoints_Town04

        # run simulation
        i = 0
        perception_req = None
        perception_res = None
        while not is_done(car):
            # walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(y=-1), speed=1))
            # walker_location = walker.get_location()
            # walker_bb = [
            #     walker_location.x - walker.bounding_box.extent.x - 0.25, walker_location.y - walker.bounding_box.extent.y - 0.25,
            #     walker_location.x + walker.bounding_box.extent.x + 0.25, walker_location.y + walker.bounding_box.extent.y + 0.25
            # ]
            world.tick()
            car.localize()
            if i % PERCEPTION_LATENCY == 0:
                car.perceive()
                # car.car.obs = obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs + [walker_bb])

                # process perception response on the car
                if perception_res: car.car.obs = perception_res; print("car Received perception response")
                plt.cla()
                plt.imshow(car.car.obs.obs, cmap='gray')
                plt.savefig('obs_map.png')

                # process perception request on server
                if perception_req:
                    print("server Received perception request")
                    perception_res = obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs)
                    mask_obstacle_map(perception_res, perception_req.x - perception_res.min_x, perception_req.y - perception_res.min_y)

                # send next request
                print("car sending request")
                perception_req = car.car.cur

            if i % 15 == 0 and car.car.cur.distance(car.car.destination) > 10:
                car.plan()
            car.run_step()
            car.process_recording_frames()
            i += 1
            # town04_spectator_follow(world, car)

        iou = car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
    finally:
        recording_cam.destroy()
        car.destroy()
        for parked_car in parked_cars:
            parked_car.destroy()
        for traffic_cone in traffic_cones:
            traffic_cone.destroy()
        # walker.destroy()

def main():
    try:
        client = load_client()

        # load map
        world = town04_load(client)

        # load spectator
        town04_spectator_bev(world)

        # load recording file
        recording_file = iio.imopen('./test.mp4', 'w', plugin='pyav')
        recording_file.init_video_stream('vp9', fps=30)

        # run scenarios
        ious = []
        for destination_parking_spot, parked_spots in SCENARIOS:
            print(f'Running scenario: destination={destination_parking_spot}, parked_spots={parked_spots}')
            run_scenario(world, destination_parking_spot, parked_spots, ious, recording_file)

        # graph ious
        plt.clf()
        plt.boxplot(ious, positions=[1], vert=True, patch_artist=True, widths=0.5,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
        x_scatter = np.random.normal(loc=0.5, scale=0.05, size=len(ious))
        plt.scatter(x_scatter, ious, color='darkblue', alpha=0.6, label='Data Points')
        plt.xticks([1], ['IOU Values'])  # Set x-ticks at the boxplot
        plt.title('Parking IOU Values')
        plt.ylabel('IOU Value')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig('iou_boxplot.png')

    except KeyboardInterrupt:
        print('stopping simulation')
    
    finally:
        recording_file.close()
        world.tick()

if __name__ == '__main__':
    main()