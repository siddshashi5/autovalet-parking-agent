from v2_experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spectator_follow,
    town04_get_drivable_graph
)

# For lane waypoint hack
from parking_position import (
    parking_lane_waypoints_Town04
)

import numpy as np
import matplotlib.pyplot as plt

SCENARIOS = [
    (17, [16, 18]),
    (25, [24, 26]),
    (18, [17, 19]),
    (19, [18, 20]),
    (35, [34, 36]),
]

def run_scenario(world, destination_parking_spot, parked_spots, ious):
    try:
        # load parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(world, parked_spots, destination_parking_spot)

        # load car
        car = town04_spawn_ego_vehicle(world, destination_parking_spot)

        # HACK: enable perfect perception of parked cars
        car.car.obs = parked_cars_bbs

        # HACK: set lane waypoints to guide parking in adjacent lanes
        car.car.lane_wps = parking_lane_waypoints_Town04

        # run simulation
        while not is_done(car):
            world.tick()
            car.run_step()
            # town04_spectator_follow(world, car)

        iou = car.iou()
        ious.append(iou)
        print(f'IOU: {iou}')
    finally:
        car.destroy()
        for parked_car in parked_cars:
            parked_car.destroy()

def main():
    try:
        client = load_client()

        # load map
        world = town04_load(client)

        # load spectator
        town04_spectator_bev(world)

        # run scenarios
        ious = []
        for destination_parking_spot, parked_spots in SCENARIOS:
            print(f'Running scenario: destination={destination_parking_spot}, parked_spots={parked_spots}')
            run_scenario(world, destination_parking_spot, parked_spots, ious)

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

if __name__ == '__main__':
    main()