import random
import carla
import numpy as np
import matplotlib.pyplot as plt
import networkx
from skimage.draw import line
from parking_position import (
    parking_vehicle_locations_Town04,
    parking_vehicle_rotation, 
    player_location_Town04,
    town04_bound 
)
from v2 import CarlaCar, Mode

HOST = '127.0.0.1'
PORT = 2000
DEBUG = True
EGO_VEHICLE = 'vehicle.tesla.model3'
PARKED_VEHICLES = [
    'vehicle.mercedes.coupe_2020',
    'vehicle.dodge.charger_2020',
    'vehicle.ford.mustang',
    'vehicle.jeep.wrangler_rubicon',
    'vehicle.lincoln.mkz_2017',
    'vehicle.audi.a2',
    'vehicle.bmw.grandtourer',
    'vehicle.chevrolet.impala',
    'vehicle.mini.cooper_s',
    'vehicle.toyota.prius'
]

def load_client():
    print(f"starting simulation on {HOST}:{PORT}")
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    return client

def is_done(car):
    if car.car.mode == Mode.FAILED:
        print("FAILED")
    return car.car.mode == Mode.PARKED or car.car.mode == Mode.FAILED

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
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    client.reload_world(False)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    return world

def town04_spawn_ego_vehicle(world, destination_parking_spot):
    destination_parking_spot_loc = parking_vehicle_locations_Town04[destination_parking_spot]
    blueprint = world.get_blueprint_library().filter(EGO_VEHICLE)[0]
    return CarlaCar(world, blueprint, player_location_Town04, destination_parking_spot_loc, approximate_bb_from_center(destination_parking_spot_loc), debug=DEBUG)

def town04_spawn_parked_cars(world, spawn_points, skip, num_random_cars):
    blueprints = world.get_blueprint_library().filter('vehicle.*.*')
    blueprints = [bp for bp in blueprints if bp.id in PARKED_VEHICLES]
    parked_cars = []
    parked_cars_bbs = []

    random_spawn_points = random.sample(range(len(parking_vehicle_locations_Town04)), num_random_cars)
    new_spawn_points = spawn_points.copy()
    for spawn_point in random_spawn_points:
        if spawn_point != skip and spawn_point not in spawn_points:
            new_spawn_points.append(spawn_point)
    spawn_points = new_spawn_points

    for i in spawn_points:
        spawn_point = parking_vehicle_locations_Town04[i]
        npc_transform = carla.Transform(spawn_point, rotation=random.choice(parking_vehicle_rotation))
        npc_bp = random.choice(blueprints)
        if npc_bp.has_attribute('color'):
            color = random.choice(npc_bp.get_attribute('color').recommended_values)
            npc_bp.set_attribute('color', color)
        npc = world.try_spawn_actor(npc_bp, npc_transform)
        if npc is None:
            parked_cars_bbs.append(approximate_bb_from_center(spawn_point))
            continue
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

def town04_get_grid(world):
    x_size = town04_bound["x_max"] - town04_bound["x_min"] + 1
    y_size = town04_bound["y_max"] - town04_bound["y_min"] + 1

    grid = np.zeros((x_size, y_size), dtype=int)

    vehicles = world.get_actors().filter('vehicle.*')

    for x in range(town04_bound["x_min"], town04_bound["x_max"] + 1):
        for y in range(town04_bound["y_min"], town04_bound["y_max"] + 1):
            is_drivable = True
            point_location = carla.Location(x=x, y=y, z=0.3)

            for vehicle in vehicles:
                bounding_box = vehicle.bounding_box
                vehicle_transform = vehicle.get_transform()
                
                # Check if the point is within the vehicle's bounding box
                if bounding_box.contains(point_location, vehicle_transform):
                    is_drivable = False
                    break

            if is_drivable:
                x_index = x - town04_bound["x_min"]
                y_index = y - town04_bound["y_min"]
                grid[x_index, y_index] = 1
    
    # plt.imshow(grid, cmap='gray', origin='lower')
    # plt.colorbar(label="Drivable (1) / Non-Drivable (0)")
    # plt.title("Drivable Area in Town04 Parking Lot")
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.show()

    return grid

def town04_get_drivable_graph(world, threshold=0.7, step=5):
    grid = town04_get_grid(world)

    drivable_grid = grid > threshold

    # Create graph and add nodes for drivable regions
    G = networkx.Graph()
    for y in range(0, drivable_grid.shape[0], step):
        for x in range(0, drivable_grid.shape[1], step):
            if drivable_grid[y, x]:
                G.add_node((x, y, 0.3))
    
    # Connect neighbors for each drivable node
    direction_vectors = [(-step, 0, 0), (step, 0, 0), (0, -step, 0), (0, step, 0), 
                         (-step, -step, 0), (step, step, 0 ), (-step, step, 0), (step, -step, 0)]
    for node in G.nodes:
        x, y, z = node
        neighbors = [(x+dx, y+dy, z + dz) for dx, dy, dz in direction_vectors]
        for nx, ny, nz in neighbors:
            if (nx, ny, nz) in G.nodes and is_path_drivable(x, y, nx, ny, drivable_grid):
                G.add_edge((x, y, z), (nx, ny, nz))

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(grid, cmap="gray", origin="upper")
    
    # # Draw the graph on top of the matrix
    # pos = {node: (node[0], node[1]) for node in G.nodes}  # Use only x and y for plotting
    # networkx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color="blue")
    # networkx.draw_networkx_edges(G, pos, ax=ax, edge_color="red", width=1)

    # plt.title("Graph of Waypoints on Drivable Map")
    # plt.savefig("drivable_graph_town04.jpg", dpi=300)
    # plt.show()

    return G

def is_path_drivable(x1, y1, x2, y2, drivable_grid):
    """Checks if the path between two points is within drivable regions."""
    rr, cc = line(y1, x1, y2, x2)  # Generate points on the line between nodes
    return np.all(drivable_grid[rr, cc])  # Check if all points on the line are drivable