import carla
import random
from parking_position import player_location_Town04, parking_vehicle_locations_Town04, parking_vehicle_rotation
from v2 import CarlaCar

HOST = '127.0.0.1'
PORT = 2000
DEBUG = True
EGO_VEHICLE = 'vehicle.tesla.model3'
PARKED_VEHICLES = [
    'vehicle.mercedes.coupe_2020',
    'vehicle.dodge.charger_2020',
    'vehicle.ford.mustang',
    'vehicle.jeep.wrangler_rubicon',
    'vehicle.lincoln.mkz_2017'
]

def main():
    try:
        print(f"starting simulation on {HOST}:{PORT}")
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)

        # load map
        world = client.load_world('Town04_Opt')
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        # load spectator
        spectator_location = carla.Location(x=285.9, y=-210.73, z=40)
        spectator_rotation = carla.Rotation(pitch=-90.0)
        world.get_spectator().set_transform(carla.Transform(spectator_location, spectator_rotation))

        # load car
        destination_parking_spot = 25
        destination_parking_spot_loc = parking_vehicle_locations_Town04[destination_parking_spot]
        destination_parking_spot_loc.y -= 0.5
        blueprint = world.get_blueprint_library().filter(EGO_VEHICLE)[0]
        car = CarlaCar(world, blueprint, player_location_Town04, destination_parking_spot_loc, debug=DEBUG)

        # load parked cars
        blueprints = world.get_blueprint_library().filter('vehicle.*.*')
        blueprints = [bp for bp in blueprints if bp.id in PARKED_VEHICLES]
        spawn_points = [24, 26]
        parked_cars = []
        parked_cars_bbs = []

        for i in spawn_points:
            spawn_point = parking_vehicle_locations_Town04[i]

            npc_transform = carla.Transform(spawn_point, rotation=random.choice(parking_vehicle_rotation))
            npc_bp = random.choice(blueprints)
            npc = world.spawn_actor(npc_bp, npc_transform)
            npc.set_simulate_physics(False)
            parked_cars.append(npc)
            parked_cars_bbs.append([
                spawn_point.x - npc.bounding_box.extent.x, spawn_point.y - npc.bounding_box.extent.y,
                spawn_point.x + npc.bounding_box.extent.x, spawn_point.y + npc.bounding_box.extent.y
            ])

        for i, loc in enumerate(parking_vehicle_locations_Town04):
            if i == destination_parking_spot or i in spawn_points: continue
            parked_cars_bbs.append([
                loc.x - 2.4, loc.y - 0.96,
                loc.x + 2.4, loc.y + 0.96
            ])

        # HACK: enable perfect perception of parked cars
        car.car.obs = parked_cars_bbs

        # run simulation
        while True:
            world.wait_for_tick()
            car.run_step()

    except KeyboardInterrupt:
        print("stopping simulation")
    finally:
        car.destroy()
        for parked_car in parked_cars:
            parked_car.destroy()

if __name__ == '__main__':
    main()