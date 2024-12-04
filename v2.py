from math import sqrt
from enum import Enum
from typing import Tuple
from queue import Queue
import carla
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
import cv2
from hybrid_a_star.hybrid_a_star import hybrid_a_star_planning as hybrid_astar
from v2_controller import VehiclePIDController

def kmph_to_mps(speed): return speed/3.6
def mps_to_kmph(speed): return speed*3.6

DESTINATION_THRESHOLD = 0.2
REPLAN_THRESHOLD = 2
LOOKAHEAD = 3
TRAJECTORY_EXTENSION = 5
MAX_ACCELERATION = 1
MAX_SPEED = kmph_to_mps(10)
MIN_SPEED = kmph_to_mps(2)
STOP_CONTROL = carla.VehicleControl(brake=1.0)

class Mode(Enum):
    DRIVING = 0
    PARKED = 1
    FAILED = 2

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
    def __init__(self, direction: Direction, x: float, y: float, speed: float, angle: float):
        self.direction = direction
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle

    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def offset(self, sign: int = 1):
        return TrajectoryPoint(self.direction, self.x + 1.6*sign*np.cos(self.angle), self.y + 1.6*sign*np.sin(self.angle), self.speed, self.angle)

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
        return []
    result_x = hybrid_astar_path.x_list
    result_y = hybrid_astar_path.y_list
    result_yaw = hybrid_astar_path.yaw_list
    result_direction = hybrid_astar_path.direction_list

    # generate trajectory points
    trajectory = [TrajectoryPoint(Direction.FORWARD if d else Direction.REVERSE, x, y, MIN_SPEED, yaw) for x, y, yaw, d in zip(result_x, result_y, result_yaw, result_direction)]
    trajectory[0].speed = cur.speed
    trajectory[0].angle = cur.angle
    refine_trajectory(trajectory)
    
    return trajectory

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
        self.recording_file = None
        self.has_recorded_segment = False
        self.frames = Queue()

        self.debug = debug
        if debug:
            self.debug_init(spawn_point, destination)

    def run_step(self):
        self.actor.apply_control(self.car.run_step())

        if self.debug:
            self.debug_step()
        
    def init_recording(self, recording_file):
        self.recording_file = recording_file
        world = self.world
        actor = self.actor
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(1080))
        cam_bp.set_attribute('image_size_y', str(720))
        cam_bp.set_attribute('fov', str(90))
        cam_location = actor.get_transform().transform(carla.Location(x=-10, z=5))
        cam_rotation = actor.get_transform().rotation
        cam_rotation.pitch -= 20
        cam_transform = carla.Transform(cam_location, cam_rotation)
        cam = world.spawn_actor(cam_bp, cam_transform, attach_to=actor, attachment_type=carla.AttachmentType.Rigid)
        cam.listen(lambda image: self.frames.put(image))
        return cam
    
    def process_recording_frames(self):
        while not self.frames.empty():
            if self.has_recorded_segment and self.car.mode == Mode.PARKED:
                return
            image = self.frames.get()
            self.has_recorded_segment = False
            recording_file = self.recording_file
            data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            data = data[:, :, :3].copy()
            data = cv2.putText(
                data,
                "autonomous, 3x speed",
                (20, image.height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2
            )
            data = cv2.putText(
                data,
                "IOU: {:.2f}".format(self.iou()),
                (image.width - 175, image.height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2
            )
            recording_file.write_frame(data, pixel_format='bgr24')
            if self.car.mode == Mode.PARKED:
                self.has_recorded_segment = True
                for _ in range(15):
                    recording_file.write_frame(data, pixel_format='bgr24')
    
    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

    def debug_step(self):
        cur = self.car.cur
        self.world.debug.draw_string(carla.Location(x=cur.x, y=cur.y), 'X', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
        future_wp = self.car.trajectory[min(self.car.ti + LOOKAHEAD, len(self.car.trajectory) - 1)]
        self.world.debug.draw_string(carla.Location(x=future_wp.x, y=future_wp.y), 'o', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=0.1, persistent_lines=True)
        for i, loc in enumerate(self.car.trajectory):
            if i == min(self.car.ti + LOOKAHEAD, len(self.car.trajectory) - 1):
                color = carla.Color(r=0, g=0, b=255)
            else:
                color = carla.Color(r=255, g=0, b=0)
            self.world.debug.draw_string(carla.Location(x=loc.x, y=loc.y), 'o', draw_shadow=False, color=color, life_time=1.0, persistent_lines=True)

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
        if self.debug:
            self.world.debug.draw_string(carla.Location(x=car_vertices[0][0], y=car_vertices[0][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=car_vertices[1][0], y=car_vertices[1][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=car_vertices[2][0], y=car_vertices[2][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=car_vertices[3][0], y=car_vertices[3][1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1, persistent_lines=True)

            self.world.debug.draw_string(carla.Location(x=destination_vertices[0][0], y=destination_vertices[0][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=destination_vertices[1][0], y=destination_vertices[1][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=destination_vertices[2][0], y=destination_vertices[2][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
            self.world.debug.draw_string(carla.Location(x=destination_vertices[3][0], y=destination_vertices[3][1]), 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)

        car_polygon = Polygon(car_vertices)
        destination_polygon = Polygon(destination_vertices)
        iou = car_polygon.intersection(destination_polygon).area / car_polygon.union(destination_polygon).area
        return iou

class Car():
    def __init__(self, destination: Tuple[float, float], gnss_sensor: CarlaGnssSensor):
        self.cur = TrajectoryPoint(Direction.FORWARD, 0, 0, 0, 0)
        self.obs: list[list[float]] = []
        self.destination = TrajectoryPoint(Direction.FORWARD, destination[0], destination[1], MIN_SPEED, 0).offset(-1)
        self.controller = VehiclePIDController({'K_P': 2, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 0.5, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})
        self.gnss_sensor = gnss_sensor
        self.trajectory: list[TrajectoryPoint] = []
        self.ti = 0
        self.mode = Mode.DRIVING

    def perceive(self):
        self.cur.x, self.cur.y = self.gnss_sensor.get_location()
        self.cur.speed = self.gnss_sensor.get_speed()
        self.cur.angle = self.gnss_sensor.get_heading()
        self.cur = self.cur.offset(-1)

    def plan(self):
        cur = self.cur
        destination = self.destination
        distance_to_destination = cur.distance(destination)
        if self.mode == Mode.PARKED or distance_to_destination < DESTINATION_THRESHOLD and self.ti >= len(self.trajectory) - TRAJECTORY_EXTENSION - 1:
            self.mode = Mode.PARKED
            return STOP_CONTROL

        # replan trajectory if needed
        trajectory = self.trajectory
        should_extend = len(trajectory) == 0
        should_fix = len(trajectory) > 0 and cur.distance(trajectory[self.ti]) > REPLAN_THRESHOLD
        if should_extend or should_fix:
            new_trajectory = plan_hybrid_astar(cur, destination, self.obs)

            if not new_trajectory:
                destination.angle += np.pi
                destination = self.destination = destination.offset(-2)
                new_trajectory = plan_hybrid_astar(cur, destination, self.obs)

            if new_trajectory:
                for i in range(1, TRAJECTORY_EXTENSION+1):
                    new_trajectory.append(destination.offset(i/3))
                self.ti = 0
                trajectory = self.trajectory = new_trajectory
            else:
                self.mode = Mode.FAILED; return STOP_CONTROL
            # plot_trajectory(trajectory)

        # find next waypoint
        ti = self.ti
        wp = trajectory[ti]
        wp_dist = cur.distance(wp)
        for i in range(ti + 1, len(trajectory)):
            if cur.distance(trajectory[i]) > wp_dist:
                break
            ti = i
            wp = trajectory[i]
            wp_dist = cur.distance(wp)
        self.ti = ti

        # find lookahead waypoint
        wp = trajectory[ti]
        future_wp = wp
        for i in range(ti + 1, ti + LOOKAHEAD + 1):
            if i >= len(trajectory):
                break
            new_dist = cur.distance(trajectory[i])
            if new_dist < wp_dist:
                break
            future_wp = trajectory[i]
            wp_dist = new_dist

        cur.direction = wp.direction
        ctrl = self.controller.run_step(
            mps_to_kmph(cur.speed),
            mps_to_kmph(wp.speed),
            cur,
            future_wp,
            wp.direction == Direction.REVERSE
        )
        return ctrl

    def run_step(self):
        self.perceive()
        return self.plan()