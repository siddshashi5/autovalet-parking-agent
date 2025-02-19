from v2_experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spawn_traffic_cones,
    town04_spawn_walkers,
    update_walkers,
    obstacle_map_from_bbs,
    clear_obstacle_map,
    union_obstacle_map,
    mask_obstacle_map,
    clear_destination_obstacle_map,
    cleanup,
    DELTA_SECONDS
)

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

SCENARIOS = [
    # (17, [16, 18]),
    # (18, [17, 19]),
    # (19, [18, 20]),
    (20, [19, 21]),
    # (21, [20, 22]),
    # (22, [21, 23]),
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
NUM_RANDOM_CARS = 50
PERCEPTION_LATENCIES = [100, 300] # ms

def run_scenario(world, destination_parking_spot, parked_spots, latency, ious, accelerations, recording_file):
    try:
        # load parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(world, parked_spots, destination_parking_spot, NUM_RANDOM_CARS)

        # spawn traffic cones
        traffic_cones, traffic_cone_bbs = town04_spawn_traffic_cones(world, [
            (284, -230),
            (287, -225),
        ])

        # spawn walker
        walkers, walker_bbs = town04_spawn_walkers(world, [
            # (285, -232),
        ])

        # tick world to load actors
        world.tick()

        # load car
        car = town04_spawn_ego_vehicle(world, destination_parking_spot)
        recording_cam = car.init_recording(recording_file)

        # HACK: enable perfect perception of parked cars
        car.car.obs = clear_obstacle_map(obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs + walker_bbs))
        # visualize 1 hot map
        plt.cla()
        plt.imshow(car.car.obs.obs, cmap='gray')
        plt.savefig('obs_map.png')

        # run simulation
        i = 0
        perception_req = None
        perception_res = None
        world.tick()
        while not is_done(car):
            walker_bbs = update_walkers(walkers)
            world.tick()
            car.localize()
            accelerations.append(car.actor.get_acceleration().length())
            if latency == 0:
                car.perceive()
                car.car.obs = union_obstacle_map(
                    car.car.obs,
                    mask_obstacle_map(
                        obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs + walker_bbs),
                        car.car.cur.x,
                        car.car.cur.y
                    ),
                )
            elif i % int(latency / 1000 / DELTA_SECONDS) == 0:
                # car.car.obs = clear_obstacle_map(car.car.obs)
                # print('perception snp: ', world.get_snapshot().find(car.actor.id).get_transform().location)
                # print('perception cur: ', car.actor.get_transform().location)
                # print('perception cur: ', world.get_snapshot().timestamp.elapsed_seconds)
                # car.car.perceive(world.get_snapshot().find(car.actor.id).get_transform())
                car.perceive()
                car.car.obs.obs[75, :] = 999
                car.car.obs.obs[140, :] = 999
                car.car.obs.obs[:, 140] = 999
                clear_destination_obstacle_map(car.car.obs, destination_parking_spot)
                # TEMP: perfect perception for now
                # car.car.obs.obs[np.where(obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs + walker_bbs).obs == 1)] = 999
                cpy = car.car.obs.probs().copy()
                cpy[np.where(obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs + walker_bbs).obs == 1)] = 1
                plt.cla()
                plt.imshow(cpy[::-1], cmap='gray', vmin=0.0, vmax=1.0)
                plt.savefig('obs_map.png')

                # # process perception response on the car
                # if perception_res: car.car.obs = perception_res

                # # process perception request on server
                # if perception_req:
                #     perception_res = union_obstacle_map(
                #         car.car.obs,
                #         mask_obstacle_map(
                #             obstacle_map_from_bbs(parked_cars_bbs + traffic_cone_bbs + walker_bbs),
                #             perception_req.x,
                #             perception_req.y
                #         ),
                #     )

                # send next request
                perception_req = car.car.cur

            if i % 5 == 0:
                car.plan()
            car.run_step()
            car.process_recording_frames(latency=latency)
            i += 1

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
        for walker in walkers:
            walker.destroy()

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
        latency_data = []
        for latency in PERCEPTION_LATENCIES:
            ious = []
            accelerations = []
            print(f'Running scenarios for latency: {latency}ms')
            for destination_parking_spot, parked_spots in SCENARIOS:
                print(f'Running scenario: destination={destination_parking_spot}, parked_spots={parked_spots}')
                run_scenario(world, destination_parking_spot, parked_spots, latency, ious, accelerations, recording_file)
            latency_data.append((latency, ious, accelerations))
            
        # scatter ious for each latency value
        plt.clf()
        for latency, ious, accelerations in latency_data:
            jerks = np.diff(accelerations)
            x_scatter = np.random.normal(loc=latency, scale=0.05, size=len(jerks))
            plt.scatter(x_scatter, jerks, alpha=0.6, label=f'{latency}ms')
        plt.title('Parking IOU Values')
        plt.xticks(PERCEPTION_LATENCIES, [f'{latency}ms' for latency in PERCEPTION_LATENCIES])
        plt.xlabel('Perception Latency')
        plt.ylabel('IOU Value')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('iou_scatter.png')

        # graph ious
        # plt.clf()
        # plt.boxplot(ious, positions=[1], vert=True, patch_artist=True, widths=0.5,
        #     boxprops=dict(facecolor='lightblue', color='blue'),
        #     medianprops=dict(color='red'))
        # x_scatter = np.random.normal(loc=0.5, scale=0.05, size=len(ious))
        # plt.scatter(x_scatter, ious, color='darkblue', alpha=0.6, label='Data Points')
        # plt.xticks([1], ['IOU Values'])  # Set x-ticks at the boxplot
        # plt.title('Parking IOU Values')
        # plt.ylabel('IOU Value')
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.legend()
        # plt.savefig('iou_boxplot.png')

    except KeyboardInterrupt:
        print('stopping simulation')
    
    finally:
        recording_file.close()
        world.tick()
        cleanup()

if __name__ == '__main__':
    main()