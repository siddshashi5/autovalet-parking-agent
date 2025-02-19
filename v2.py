from math import sqrt
from enum import Enum
from typing import Tuple
from queue import Queue
import json
import carla
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import cv2
from scipy.spatial.transform import Rotation
from hybrid_a_star.hybrid_a_star import hybrid_a_star_planning as hybrid_a_star
from hybrid_a_star.car import rectangle_check, BUBBLE_DIST, BUBBLE_R
from fisheye_camera import FisheyeCamera, EquidistantProjection, StereographicProjection
from v2_perception import run_perception_model
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
        return TrajectoryPoint(self.direction, self.x + 1.4055*sign*np.cos(self.angle), self.y + 1.4055*sign*np.sin(self.angle), self.speed, self.angle)

class ObstacleMap():
    def __init__(self, min_x: int, min_y: int, obs: np.array):
        self.min_x = min_x
        self.min_y = min_y
        self.obs = obs

    def transform_coord(self, x: float, y: float):
        x = int((x - self.min_x) / 0.25)
        y = int((y - self.min_y) / 0.25)
        return x, y

    def inverse_transform_coord(self, x: float, y: float):
        x = x * 0.25 + self.min_x
        y = y * 0.25 + self.min_y
        return x, y

    def circular_mask(self, x: float, y: float, r: float):
        x, y = self.transform_coord(x, y)
        r = int(r / 0.25)
        x_coords, y_coords = np.ogrid[:self.obs.shape[0], :self.obs.shape[1]]
        mask = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2) < r
        return mask

    def probs(self):
        return 1 - (1 / (1 + np.exp(self.obs)))

    def check_collision(self, trajectory: list[TrajectoryPoint]):
        probs = self.probs()
        mask = np.zeros_like(probs, dtype=bool)
        x_coords, y_coords = np.meshgrid(np.arange(probs.shape[0]), np.arange(probs.shape[1]))

        for wp in trajectory[:min(10, len(trajectory)-TRAJECTORY_EXTENSION):2]:
            x, y = self.transform_coord(wp.x, wp.y)
            R = np.array([
                [np.cos(-wp.angle), -np.sin(-wp.angle)],
                [np.sin(-wp.angle), np.cos(-wp.angle)]
            ])
            coords = np.stack([x_coords.flatten() - x, y_coords.flatten() - y])
            rotated = R @ coords
            rear_distance = 1.045 * 4
            front_distance = 3.856 * 4
            width = 2.0 * 4
            hits = (rotated[0] >= -rear_distance) & (rotated[0] <= front_distance) & (rotated[1] >= -width/2) & (rotated[1] <= width/2)
            mask |= hits.reshape(probs.shape[1], probs.shape[0]).T
        
        plt.cla()
        plt.imshow(mask[::-1] | np.rint(probs[::-1]).astype(bool), cmap='gray', vmin=0.0, vmax=1.0)
        plt.savefig('obs_map_mask.png')

        plt.cla()
        plt.imshow(mask[::-1] & np.rint(probs[::-1]).astype(bool), cmap='gray', vmin=0.0, vmax=1.0)
        # plt.imshow(probs[::-1], cmap='gray', vmin=0.0, vmax=1.0)
        plt.savefig('obs_map_hits.png')

        print((probs[mask] > 0.5).sum())
        return np.any(probs[mask] > 0.5)


        # for wp in trajectory:
        #     cx = wp.x + BUBBLE_DIST * np.cos(wp.angle)
        #     cy = wp.y + BUBBLE_DIST * np.sin(wp.angle)
        #     xs = []
        #     ys = []

        #     for x in range(int((cx - self.min_x - BUBBLE_R) / 0.25), int((cx - self.min_x + BUBBLE_R) / 0.25) + 1):
        #         for y in range(int((cy - self.min_y - BUBBLE_R) / 0.25), int((cy - self.min_y + BUBBLE_R) / 0.25) + 1):
        #             if 0 <= x < len(self.obs) and 0 <= y < len(self.obs[0]) and probs[x][y] > 0.5:
        #                 xs.append(x * 0.25 + self.min_x)
        #                 ys.append(y * 0.25 + self.min_y)

        #     if not xs:
        #         continue

        #     if not rectangle_check(wp.x, wp.y, wp.angle, xs, ys):
        #         return True

        # return False

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

def plan_hybrid_a_star(cur: TrajectoryPoint, destination: TrajectoryPoint, obs: ObstacleMap) -> list[TrajectoryPoint]:
    # run planner
    start = np.array([cur.x - obs.min_x, cur.y - obs.min_y, cur.angle])
    end = np.array([destination.x - obs.min_x, destination.y - obs.min_y, destination.angle])
    local_min_x = min(cur.x, destination.x) - 8
    local_max_x = max(cur.x, destination.x) + 8
    local_min_y = min(cur.y, destination.y) - 8
    local_max_y = max(cur.y, destination.y) + 8
    local_min_x, local_min_y = obs.transform_coord(local_min_x, local_min_y)
    local_max_x, local_max_y = obs.transform_coord(local_max_x, local_max_y)
    ox = []
    oy = []
    probs = obs.probs()
    probs[0, :] = 1
    probs[-1, :] = 1
    probs[:, 0] = 1
    probs[:, -1] = 1

    probs[local_min_x, :] = 1
    probs[local_max_x, :] = 1
    probs[:, local_min_y] = 1
    probs[:, local_max_y] = 1

    for coord in np.argwhere(probs > 0.5):
        ox.append(coord[0]*.25)
        oy.append(coord[1]*.25)

    hybrid_astar_path = hybrid_a_star(start, end, ox, oy, 2.0, np.deg2rad(15.0))
    if not hybrid_astar_path:
        return []
    result_x = hybrid_astar_path.x_list
    result_y = hybrid_astar_path.y_list
    result_yaw = hybrid_astar_path.yaw_list
    result_direction = hybrid_astar_path.direction_list

    # sometimes the direction list is too short
    if len(result_direction) < len(result_x):
        for _ in range(len(result_x) - len(result_direction)):
            result_direction.append(result_direction[-1])

    # generate trajectory points
    trajectory = [TrajectoryPoint(Direction.FORWARD if d else Direction.REVERSE, x + obs.min_x, y + obs.min_y, MIN_SPEED, yaw) for x, y, yaw, d in zip(result_x, result_y, result_yaw, result_direction)]
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

class CarlaCameraSensor():
    def __init__(self, actor, world):
        self.actor = actor
        self.cameras = {}
        with open('./v2_camera_config.json', 'r') as file:
            calibration = json.load(file)
            for cam_name in calibration:
                cam_config = calibration[cam_name]
                x = cam_config['spawn_point']['x']
                y = -cam_config['spawn_point']['y']
                z = cam_config['spawn_point']['z']
                roll = cam_config['spawn_point']['roll']
                pitch = cam_config['spawn_point']['pitch']
                yaw = cam_config['spawn_point']['yaw']
                quat = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()
                roll, pitch, yaw = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
                pitch = -pitch
                yaw = -yaw
                width = cam_config['x_size']
                height = cam_config['y_size']
                fx = cam_config['f_x']
                fy = cam_config['f_y']
                cx = cam_config['c_x']
                cy = cam_config['c_y']
                max_angle = cam_config['max_angle']
                cam = FisheyeCamera(
                    parent_actor=actor, camera_model=EquidistantProjection, width=width, height=height, tick=0.0,
                    x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, fx=fx, fy=fy, cx=cx, cy=cy,
                    k0=0.0, k1=0.0, k2=0.0, k3=0.0, k4=0.0,
                    max_angle=max_angle, camera_type='sensor.camera.rgb'
                )

                # cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
                # cam_bp.set_attribute('image_size_x', str(cam_config['image_width']))
                # cam_bp.set_attribute('image_size_y', str(cam_config['image_height']))
                # fov = 2 * np.arctan(cam_config['image_width'] / (2 * cam_config['intrinsic']['fx'])) * 180 / np.pi
                # cam_bp.set_attribute('fov', str(fov))

                # rotation = Rotation.from_matrix(cam_config['rotation'])
                # pitch, yaw, roll = rotation.as_euler('YZX', degrees=True)
                # rotation = carla.Rotation(pitch, yaw, roll)

                # translation = cam_config['translation']
                # location = carla.Location(translation[0], translation[1], translation[2])

                # transform = carla.Transform(location, rotation)
                # cam = world.spawn_actor(cam_bp, transform, attach_to=actor, attachment_type=carla.AttachmentType.Rigid)
                self.cameras[cam_name] = cam

    def get_images(self):
        images = {}
        for cam_name in self.cameras:
            self.cameras[cam_name].create_fisheye_image()
            images[cam_name] = self.cameras[cam_name].image
        return images

    def destroy(self):
        for cam in self.cameras.values():
            cam.destroy()

class CarlaCar():
    def __init__(self, world, blueprint, spawn_point, destination, destination_bb, debug=False):
        self.world = world
        self.actor = world.spawn_actor(blueprint, spawn_point)
        self.gnss_sensor = CarlaGnssSensor(self.actor)
        self.camera_sensor = CarlaCameraSensor(self.actor, world)
        self.car = Car((destination.x, destination.y), self.gnss_sensor, self.camera_sensor)
        self.destination_bb = destination_bb
        self.recording_file = None
        self.has_recorded_segment = False
        self.frames = Queue()

        self.debug = debug
        if debug:
            self.debug_init(spawn_point, destination)

    def localize(self): self.car.localize()
    def perceive(self): self.car.perceive()
    def plan(self): self.car.plan()

    def run_step(self):
        self.actor.apply_control(self.car.control())
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
        # for cam_name in self.camera_frames:
        #     self.camera_sensor.cameras[cam_name].listen(lambda image, cam_name=cam_name: self.camera_frames[cam_name].put(image))

        return cam
    
    def process_recording_frames(self, latency=None):
        while not self.frames.empty():
            if self.has_recorded_segment and self.car.mode == Mode.PARKED:
                return
            image = self.frames.get()
            camera_images = self.camera_sensor.get_images()
            self.has_recorded_segment = False
            recording_file = self.recording_file

            data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            data = data[:, :, :3].copy()
            data = data[:, :, ::-1]
            cams_data = np.zeros((180, image.width, 3))

            for cam_index, cam_name in enumerate(camera_images):
                cam_data = camera_images[cam_name].copy()
                # if cam_name == 'rgb_left':
                #     cam_data = cv2.flip(cam_data, 0)
                cam_data = cv2.resize(cam_data, (int(image.width / 4), int(image.height / 4)))
                cams_data[:cam_data.shape[0], (cam_index * cam_data.shape[1]):((cam_index + 1) * cam_data.shape[1])] = cam_data
            
            data = np.concatenate((data, cams_data), axis=0)

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
            if latency:
                data = cv2.putText(
                    data,
                    "latency: {}ms".format(latency) if latency else "",
                    (image.width - 275, image.height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2
                )
            recording_file.write_frame(data, pixel_format='rgb24')
            if self.car.mode == Mode.PARKED:
                self.has_recorded_segment = True
                for _ in range(15):
                    recording_file.write_frame(data, pixel_format='rgb24')
    
    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

    def debug_step(self):
        cur = self.car.cur
        self.world.debug.draw_string(carla.Location(x=cur.x, y=cur.y), 'X', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=0.1, persistent_lines=True)
        for loc in self.car.trajectory:
            self.world.debug.draw_string(carla.Location(x=loc.x, y=loc.y), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)

    def destroy(self):
        self.actor.destroy()
        self.camera_sensor.destroy()

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
    def __init__(self, destination: Tuple[float, float], gnss_sensor: CarlaGnssSensor, camera_sensor: CarlaCameraSensor):
        self.cur = TrajectoryPoint(Direction.FORWARD, 0, 0, 0, 0)
        self.obs: ObstacleMap = []
        self.destination = TrajectoryPoint(Direction.FORWARD, destination[0], destination[1], MIN_SPEED, 0).offset(-1)
        self.controller = VehiclePIDController({'K_P': 2, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 0.5, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})
        self.gnss_sensor = gnss_sensor
        self.camera_sensor = camera_sensor
        self.trajectory: list[TrajectoryPoint] = []
        self.ti = 0
        self.mode = Mode.DRIVING

    def localize(self):
        self.cur.x, self.cur.y = self.gnss_sensor.get_location()
        self.cur.speed = self.gnss_sensor.get_speed()
        self.cur.angle = self.gnss_sensor.get_heading()
        self.cur = self.cur.offset(-1)

    def perceive(self):
        imgs = self.camera_sensor.get_images()
        # check if any are nonzero
        if np.nonzero(imgs['rgb_front'])[0].size == 0:
            return
        # print('used: ', self.cur.x, self.cur.y)
        cur_offset = self.cur
        cur_x = cur_offset.x
        cur_y = cur_offset.y
        cur_angle = cur_offset.angle
        # cur_x = actual.location.x
        # cur_y = actual.location.y
        # cur_angle = np.deg2rad(actual.rotation.yaw)
        occ = run_perception_model(cur_x, -cur_y, -cur_angle, [
            imgs['rgb_front'],
            imgs['rgb_left'],
            imgs['rgb_rear'],
            imgs['rgb_right']
        ])
        if occ is not None:
            updates = np.zeros_like(self.obs.obs)
            occluded = np.zeros_like(self.obs.obs, dtype=bool)
            # cur_x_min, cur_y_min = self.obs.transform_coord(self.cur.x - 8, self.cur.y - 8)
            # cur_x_max, cur_y_max = self.obs.transform_coord(self.cur.x + 8, self.cur.y + 8)
            # cur_x_min = max(0, cur_x_min)
            # cur_y_min = max(0, cur_y_min)
            # cur_x_max = min(len(self.obs.obs), cur_x_max)
            # cur_y_max = min(len(self.obs.obs[0]), cur_y_max)
            radius = 9
            mask = self.obs.circular_mask(cur_x, cur_y, radius)
            # updates[mask] = np.log(0.45 / (1 - 0.45))
            # updates[mask] = 0
            for point in occ:
                reflected_point = np.array([point[0], -point[1]])
                rotation_matrix = np.array([
                    [np.cos(cur_angle), -np.sin(cur_angle)],
                    [np.sin(cur_angle), np.cos(cur_angle)]
                ])
                rotated_point = np.dot(rotation_matrix, reflected_point)
                if np.linalg.norm(rotated_point) > radius:
                    continue
                absolute_point = rotated_point + np.array([cur_x, cur_y])
                x, y = self.obs.transform_coord(absolute_point[0], absolute_point[1])
                if 0 <= x < len(updates) and 0 <= y < len(updates[0]):
                    updates[x][y] += np.log(0.65 / (1 - 0.65))
                    # dir_vec = rotated_point / np.linalg.norm(rotated_point)
                    # for dist in range(1, radius*4):
                    #     dist = dist / 4
                    #     occluded_x, occluded_y = self.obs.transform_coord(absolute_point[0] + dist * dir_vec[0], absolute_point[1] + dist * dir_vec[1])
                    #     if 0 <= occluded_x < len(updates) and 0 <= occluded_y < len(updates[0]):
                    #         occluded[occluded_x][occluded_y] = True
                    # updates[x][y] = np.log(0.85 / (1 - 0.85))
                    # if point[-1] == 1:
                    #     updates[x][y] += np.log(0.45 / (1 - 0.45))
                    # elif point[-1] == 9:
                    #     updates[x][y] += np.log(0.55 / (1 - 0.55))
            # updates *= 0.5
            # updates = np.clip(updates, 0, 1)
            # updates[mask & occluded] *= 0.1
            # updates[mask & ~occluded] = np.log(0.45 / (1 - 0.45))
            for x, y in zip(*np.where(mask)):
                cell_point = np.array(self.obs.inverse_transform_coord(x, y)) - np.array([cur_x, cur_y])
                distance = np.linalg.norm(cell_point)
                distance_weight = 1 - (distance / radius)
                if updates[x][y] == 0:
                    updates[x][y] = np.log(0.45 / (1 - 0.45))
                updates[x][y] *= distance_weight
            self.obs.obs += updates
            # self.obs.obs = updates
            # self.obs.obs = np.clip(self.obs.obs, -5, 10)

    def plan(self):
        cur = self.cur
        destination = self.destination

        # replan trajectory if needed
        trajectory = self.trajectory
        should_extend = len(trajectory) == 0
        should_fix = len(trajectory) > 0 and cur.distance(trajectory[self.ti]) > REPLAN_THRESHOLD
        has_collision = self.obs.check_collision(trajectory[self.ti:])

        if should_extend or should_fix or has_collision:
            new_trajectory = plan_hybrid_a_star(cur, destination, self.obs)

            if not new_trajectory:
                destination.angle += np.pi
                destination = self.destination = destination.offset(-2)
                new_trajectory = plan_hybrid_a_star(cur, destination, self.obs)

            if new_trajectory:
                for i in range(1, TRAJECTORY_EXTENSION+1):
                    new_trajectory.append(destination.offset(i/3))
                self.ti = 1
                trajectory = self.trajectory = new_trajectory
            else:
                self.obs.obs[1:-1, 1:-1] = 0
                self.plan()
                # self.ti = 0
                # self.trajectory = []
                # self.mode = Mode.FAILED
        
        # decay all obstacles except the edges
        # self.obs.obs[1:-1, 1:-1] *= 0.99
    
    def control(self):
        if self.mode == Mode.FAILED: return STOP_CONTROL

        # stop if close to destination
        cur = self.cur
        destination = self.destination
        distance_to_destination = cur.distance(destination)
        if self.mode == Mode.PARKED or distance_to_destination < DESTINATION_THRESHOLD and self.ti >= len(self.trajectory) - TRAJECTORY_EXTENSION - 1:
            self.mode = Mode.PARKED
            return STOP_CONTROL

        # find next waypoint
        ti = self.ti
        trajectory = self.trajectory
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
        self.plan()
        return self.control()