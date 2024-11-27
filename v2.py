from math import sqrt
from enum import Enum
from typing import Tuple
import carla
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
import lib.frenet_optimal_trajectory_planner.FrenetOptimalTrajectory.fot_wrapper as fot
from hybrid_a_star.hybrid_a_star import hybrid_a_star_planning as hybrid_astar
from v2_controller import VehiclePIDController

def kmph_to_mps(speed): return speed/3.6
def mps_to_kmph(speed): return speed*3.6

DESTINATION_THRESHOLD = 0.2
WAYPOINT_THRESHOLD = 0.5
MAX_ACCELERATION = 1
MAX_SPEED = kmph_to_mps(10)
MIN_SPEED = kmph_to_mps(2)
STOP_CONTROL = carla.VehicleControl(brake=1.0)

class Mode(Enum):
    DRIVING = 0
    PARKING = 1
    PARKED = 2
    FAILED = 3

class Direction(Enum):
    FORWARD = 0
    REVERSE = 1

    def opposite(self):
        return Direction.FORWARD if self == Direction.REVERSE else Direction.REVERSE

def plot_trajectory(trajectory):
    x_coords = [p.x for p in trajectory]
    y_coords = [p.y for p in trajectory]
    speeds = [p.speed for p in trajectory]
    directions = [p.direction for p in trajectory]
    
    # Normalize speeds for color mapping
    norm_speeds = [speed / max(speeds) if max(speeds) > 0 else 0 for speed in speeds]
    
    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot trajectory points with arrows indicating direction
    for i in range(len(trajectory) - 1):
        ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], color=cmap(norm_speeds[i]))
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[i+1] - y_coords[i]
        ax.arrow(x_coords[i], y_coords[i], dx, dy, head_width=0.5, head_length=0.5, fc=cmap(norm_speeds[i]), ec=cmap(norm_speeds[i]))
        ax.text(x_coords[i], y_coords[i], f'{directions[i]}', fontsize=8)
    
    # Add colorbar to indicate speed
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(speeds), vmax=max(speeds)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Speed (m/s)')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Trajectory with Direction and Speed')
    ax.grid(True)
    plt.savefig('test.png')

class TrajectoryPoint():
    def __init__(self, direction: Mode, x: float, y: float, speed: float, angle: float):
        self.direction = direction
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle

    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def offset(self, sign: int = 1):
        return TrajectoryPoint(self.direction, self.x + 2*sign*np.cos(self.angle), self.y + 2*sign*np.sin(self.angle), self.speed, self.angle)

def refine_trajectory(trajectory: list[TrajectoryPoint]):
    if len(trajectory) == 0: return

    # find direction changes based on positions
    segments = [0]
    cur_direction = trajectory[0].direction
    forward_vec_x = np.cos(trajectory[0].angle)
    forward_vec_y = np.sin(trajectory[0].angle)
    if cur_direction == Direction.REVERSE:
        forward_vec_x = -forward_vec_x
        forward_vec_y = -forward_vec_y
    for i in range(len(trajectory) - 1):
        dx = trajectory[i+1].x - trajectory[i].x
        dy = trajectory[i+1].y - trajectory[i].y
        dist = sqrt(dx**2 + dy**2)
        if dist == 0:
            continue
        dot = dx * forward_vec_x + dy * forward_vec_y
        forward_vec_x = dx
        forward_vec_y = dy
        if dot < 0:
            cur_direction = trajectory[i].direction = cur_direction.opposite()
            segments.append(i)
        else:
            trajectory[i].direction = cur_direction
    if len(trajectory) > 1:
        trajectory[-1].direction = trajectory[-2].direction
    segments.append(len(trajectory))

    for segment_i in range(len(segments) - 1):
        start = segments[segment_i]
        end = segments[segment_i + 1]

        # forward pass
        for i in range(start + 1, end - 1):
            d = trajectory[i-1].distance(trajectory[i])
            trajectory[i].speed = min(MAX_SPEED, sqrt(trajectory[i-1].speed**2 + 2 * MAX_ACCELERATION * d))

        # backward pass
        for i in range(end - 2, start - 1, -1):
            d = trajectory[i-1].distance(trajectory[i])
            trajectory[i].speed = min(trajectory[i].speed, sqrt(trajectory[i+1].speed**2 + 2 * MAX_ACCELERATION * d))

def plan_hybrid_astar(cur: TrajectoryPoint, destination: TrajectoryPoint, obs: list[list[float]]) -> list[TrajectoryPoint]:
    # use Hybrid A* planner
    initial_conditions = {
        'start': np.array([cur.x, cur.y, cur.angle]),
        'end': np.array([destination.x, destination.y, destination.angle]),
        'obs': np.array(obs)
    }
    ox = []
    oy = []
    for obs in obs:
        # top and bottom
        for x in np.arange(obs[0], obs[2], .5):
            ox.append(x.item())
            oy.append(obs[1])

            ox.append(x.item())
            oy.append(obs[3])

        # left and right
        for y in np.arange(obs[1], obs[3], .5):
            ox.append(obs[0])
            oy.append(y.item())

            ox.append(obs[2])
            oy.append(y.item())

    # plot start, end, and obstacles
    hybrid_astar_path = hybrid_astar(initial_conditions['start'], initial_conditions['end'], ox, oy, 2.0, np.deg2rad(15.0))
    if not hybrid_astar_path:
        initial_conditions['end'][2] += np.pi
        hybrid_astar_path = hybrid_astar(initial_conditions['start'], initial_conditions['end'], ox, oy, 2.0, np.deg2rad(15.0))
        if not hybrid_astar_path:
            return []
    result_x = hybrid_astar_path.x_list
    result_y = hybrid_astar_path.y_list
    result_yaw = hybrid_astar_path.yaw_list

    # generate trajectory points
    trajectory = [TrajectoryPoint(cur.direction, x, y, MIN_SPEED, yaw) for x, y, yaw in zip(result_x, result_y, result_yaw)]
    trajectory[0].speed = cur.speed
    trajectory[0].angle = cur.angle
    refine_trajectory(trajectory)

    # remove points that are too close
    new_trajectory = []
    for wp in trajectory:
        if cur.distance(wp) > WAYPOINT_THRESHOLD:
            new_trajectory.append(wp)
    
    return new_trajectory

# TODO: get data from actual GNSS sensor instead of getting
# perfect vehicle data from CARLA
class CarlaGnssSensor():
    def __init__(self, actor):
        self.actor = actor

    def get_location(self) -> Tuple[float, float]:
        loc = self.actor.get_location()
        return loc.x, loc.y

    def get_speed(self) -> float:
        vel = self.actor.get_velocity()
        return vel.length()
    
    def get_heading(self):
        return np.deg2rad(self.actor.get_transform().rotation.yaw)

class CarlaCar():
    def __init__(self, world, blueprint, spawn_point, destination, destination_bb, debug=False):
        self.world = world
        self.actor = world.spawn_actor(blueprint, spawn_point)
        self.gnss_sensor = CarlaGnssSensor(self.actor)
        self.car = Car((destination.x, destination.y), self.gnss_sensor)
        self.destination_bb = destination_bb

        self.debug = debug
        if debug:
            self.debug_init(spawn_point, destination)

    def run_step(self):
        self.actor.apply_control(self.car.run_step())
        if self.debug:
            self.debug_step()

    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

    def debug_step(self):
        cur = self.car.cur
        self.world.debug.draw_string(carla.Location(x=cur.x, y=cur.y), 'X', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)
        for loc in self.car.trajectory:
            self.world.debug.draw_string(carla.Location(x=loc.x, y=loc.y), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)

    def destroy(self):
        self.actor.destroy()

    def iou(self):
        actor = self.actor
        car_transform = actor.get_transform()
        car_loc = car_transform.location
        car_angle = car_transform.rotation.yaw
        car_angle = np.deg2rad(actor.get_transform().rotation.yaw)
        car_rotation = np.array([
            [np.cos(car_angle), -np.sin(car_angle)],
            [np.sin(car_angle), np.cos(car_angle)]
        ])
        car_bb = [
            -actor.bounding_box.extent.x, -actor.bounding_box.extent.y,
            actor.bounding_box.extent.x, actor.bounding_box.extent.y
        ]
        car_vertices = [
            np.dot(car_rotation, np.array([car_bb[0], car_bb[1]])) + np.array([car_loc.x, car_loc.y]),
            np.dot(car_rotation, np.array([car_bb[0], car_bb[3]])) + np.array([car_loc.x, car_loc.y]),
            np.dot(car_rotation, np.array([car_bb[2], car_bb[3]])) + np.array([car_loc.x, car_loc.y]),
            np.dot(car_rotation, np.array([car_bb[2], car_bb[1]])) + np.array([car_loc.x, car_loc.y])
        ]
        destination_bb = self.destination_bb
        destination_vertices = [(destination_bb[0], destination_bb[1]), (destination_bb[0], destination_bb[3]), (destination_bb[2], destination_bb[3]), (destination_bb[2], destination_bb[1])]

        # Debug bounding boxes
        # self.world.debug.draw_string(carla.Location(x=car_vertices[0][0], y=car_vertices[0][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=car_vertices[1][0], y=car_vertices[1][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=car_vertices[2][0], y=car_vertices[2][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=car_vertices[3][0], y=car_vertices[3][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)

        # self.world.debug.draw_string(carla.Location(x=destination_vertices[0][0], y=destination_vertices[0][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=destination_vertices[1][0], y=destination_vertices[1][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=destination_vertices[2][0], y=destination_vertices[2][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)
        # self.world.debug.draw_string(carla.Location(x=destination_vertices[3][0], y=destination_vertices[3][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)

        car_polygon = Polygon(car_vertices)
        destination_polygon = Polygon(destination_vertices)
        iou = car_polygon.intersection(destination_polygon).area / car_polygon.union(destination_polygon).area
        return iou

class Car():
    def __init__(self, destination: Tuple[float, float], gnss_sensor: CarlaGnssSensor):
        self.cur = TrajectoryPoint(Direction.FORWARD, 0, 0, 0, 0)
        self.obs: list[list[float]] = []
        self.destination = TrajectoryPoint(Direction.FORWARD, destination[0], destination[1], 0, 0)
        self.controller = VehiclePIDController({'K_P': 2, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 0.5, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})
        self.gnss_sensor = gnss_sensor
        self.trajectory: list[TrajectoryPoint] = []
        self.mode = Mode.DRIVING

    def perceive(self):
        self.cur.x, self.cur.y = self.gnss_sensor.get_location()
        self.cur.speed = self.gnss_sensor.get_speed()
        self.cur.angle = self.gnss_sensor.get_heading()
        # TODO: get/update obstacles and lane waypoints from sensor data only,
        # adding data to graph as needed

    def plan(self):
        # TODO: handle planning for exploration phase
        # if we're at destination, stop
        # TODO: also stop if unexpected obstacle detected
        cur = self.cur
        destination = self.destination
        distance_to_destination = cur.distance(destination)
        if self.mode == Mode.PARKED or distance_to_destination < DESTINATION_THRESHOLD:
            self.mode = Mode.PARKED
            return STOP_CONTROL

        # switch to parking mode if close to destination
        if self.mode == Mode.DRIVING and distance_to_destination < 25:
            self.mode = Mode.PARKING

        # remove visited points from trajectory
        trajectory = self.trajectory
        num_to_remove = 0
        for loc in trajectory[:-1]:
            if cur.distance(loc) < WAYPOINT_THRESHOLD:
                num_to_remove += 1
            else:
                break
        if num_to_remove > 0:
            trajectory = self.trajectory = trajectory[num_to_remove:]

        # replan trajectory if needed
        should_extend = len(trajectory) == 0
        should_fix = len(trajectory) > 0 and cur.distance(trajectory[0]) > 2
        if should_extend or should_fix:
            new_trajectory = plan_hybrid_astar(cur, destination, self.obs)
            if new_trajectory:
                trajectory = self.trajectory = new_trajectory
            # plot_trajectory(trajectory)
        if not trajectory: self.mode = Mode.FAILED; return STOP_CONTROL

        # check if the next waypoint is behind us, in which case we need to reverse
        wp = trajectory[0]
        cur.direction = wp.direction
        ctrl = self.controller.run_step(
            mps_to_kmph(cur.speed),
            mps_to_kmph(wp.speed),
            cur,
            wp,
            wp.direction == Direction.REVERSE
        )
        return ctrl

    def run_step(self):
        self.perceive()
        return self.plan()