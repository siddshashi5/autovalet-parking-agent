import random
import carla
from parking_position import parking_vehicle_locations_Town04, parking_vehicle_rotation, player_location_Town04
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

def load_client():
    print(f"starting simulation on {HOST}:{PORT}")
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    return client

def approximate_bb_from_center(loc, padding=0):
    return [
        loc.x - 2.4 - padding, loc.y - 0.96,
        loc.x + 2.4 + padding, loc.y + 0.96
    ]

def town04_spectator_bev(world):
    spectator_location = carla.Location(x=285.9, y=-210.73, z=40)
    spectator_rotation = carla.Rotation(pitch=-90.0)
    world.get_spectator().set_transform(carla.Transform(spectator_location, spectator_rotation))

def town04_spectator_follow(world, car):
    spectator_rotation = car.actor.get_transform().rotation
    spectator_rotation.pitch -= 20
    spectator_transform = carla.Transform(car.actor.get_transform().transform(carla.Location(x=-10,z=5)), spectator_rotation)
    world.get_spectator().set_transform(spectator_transform)

def town04_load(client):
    world = client.load_world('Town04_Opt')
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    return world

def town04_spawn_ego_vehicle(world, destination_parking_spot):
    destination_parking_spot_loc = parking_vehicle_locations_Town04[destination_parking_spot]
    destination_parking_spot_loc.y -= 0.25
    destination_parking_spot_loc.x += 0.25
    blueprint = world.get_blueprint_library().filter(EGO_VEHICLE)[0]
    return CarlaCar(world, blueprint, player_location_Town04, approximate_bb_from_center(destination_parking_spot_loc, padding=1.2), debug=DEBUG)

def town04_spawn_parked_cars(world, spawn_points, skip):
    blueprints = world.get_blueprint_library().filter('vehicle.*.*')
    blueprints = [bp for bp in blueprints if bp.id in PARKED_VEHICLES]
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
        if i == skip or i in spawn_points: continue
        parked_cars_bbs.append(approximate_bb_from_center(loc))

    return parked_cars, parked_cars_bbs