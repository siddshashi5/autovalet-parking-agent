from v2_experiment_utils import (
    load_client,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spectator_follow,
    town04_get_drivable_grid
)

DESTINATION_PARKING_SPOT = 25
PARKED_CARS = [24, 26]

def main():
    try:
        client = load_client()

        # load map
        world = town04_load(client)

        # load spectator
        town04_spectator_bev(world)

        # load parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(world, PARKED_CARS, DESTINATION_PARKING_SPOT)

        # load car
        car = town04_spawn_ego_vehicle(world, DESTINATION_PARKING_SPOT)

        grid = town04_get_drivable_grid(world)

        # HACK: enable perfect perception of parked cars
        car.car.obs = parked_cars_bbs

        # run simulation
        while True:
            world.wait_for_tick()
            car.run_step()
            # town04_spectator_follow(world, car)

    except KeyboardInterrupt:
        print("stopping simulation")
    finally:
        car.destroy()
        for parked_car in parked_cars:
            parked_car.destroy()

if __name__ == '__main__':
    main()