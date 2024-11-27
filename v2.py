from math import sqrt
from enum import Enum
import carla
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely import Polygon
import lib.frenet_optimal_trajectory_planner.FrenetOptimalTrajectory.fot_wrapper as fot
from hybrid_a_star.hybrid_a_star import hybrid_a_star_planning as hybrid_astar
from v2_controller import VehiclePIDController

def kmph_to_mps(speed): return speed/3.6
def mps_to_kmph(speed): return speed*3.6
FOT_HYPERPARAMETERS = {
    "max_speed": kmph_to_mps(15.0),
    "max_accel": 15.0,
    "max_curvature": 15.0,
    "max_road_width_l": 3.0,
    "max_road_width_r": 3.0,
    "d_road_w": 0.5,
    "dt": 0.2,
    "maxt": 10.0,
    "mint": 5.0,
    "d_t_s": 0.5,
    "n_s_sample": 1.0,
    "obstacle_clearance": 0.2,
    "kd": 1.0,
    "kv": 0.1,
    "ka": 0.1,
    "kj": 0.1,
    "kt": 0.1,
    "ko": 0.1,
    "klat": 1.0,
    "klon": 1.0,
    "num_threads": 4,
}
DESTINATION_THRESHOLD = 0.5
WAYPOINT_THRESHOLD = 0.5
REPLAN_THRESHOLD = 5
MAX_ACCELERATION = 1
MAX_SPEED = kmph_to_mps(15)
MIN_SPEED = kmph_to_mps(1)
SLOWDOWN_CONSTANT = 10
NUM_GUIDANCE_WPS = 6
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

    def offset(self, is_reversing):
        if is_reversing:
            return TrajectoryPoint(self.direction, self.x - 2*np.cos(self.angle), self.y - 2*np.sin(self.angle), self.speed, self.angle)
        return TrajectoryPoint(self.direction, self.x + 2*np.cos(self.angle), self.y + 2*np.sin(self.angle), self.speed, self.angle)

def refine_trajectory(trajectory: list[TrajectoryPoint], direction: Direction, heading: float):
    if len(trajectory) == 0: return

    # find direction changes based on positions
    segments = [0]
    cur_direction = direction
    forward_vec_x = np.cos(heading)
    forward_vec_y = np.sin(heading)
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

def plan_fot(ps, pos, vel, guidance_wps, obs, direction, heading):
    # use FOT planner
    initial_conditions = {
        'ps': ps,
        'target_speed': MAX_SPEED,
        'pos': np.array([pos.x, pos.y]),
        'vel': np.array([vel.x, vel.y]),
        'wp': np.array(guidance_wps),
        'obs': np.array(obs)
    }
    result_x, result_y, speeds, ix, iy, result_yaw, d, s, speeds_x, speeds_y, \
        misc, costs, success = fot.run_fot(initial_conditions, FOT_HYPERPARAMETERS)
    if not success: return [], 0

    # generate trajectory points
    trajectory = [TrajectoryPoint(Direction.FORWARD, x, y, MAX_SPEED, yaw if direction == Direction.FORWARD else -yaw) for x, y, yaw in zip(result_x, result_y, result_yaw)]
    trajectory[0].speed = vel.length()
    refine_trajectory(trajectory, direction, heading)

    # remove points that are too close
    new_trajectory = []
    for wp in trajectory:
        if pos.distance(wp) > WAYPOINT_THRESHOLD:
            new_trajectory.append(wp)

    return new_trajectory, misc['s']

def plan_hybrid_astar(pos, vel, direction, heading, destination, obs):
    # use Hybrid A* planner
    initial_conditions = {
        'start': np.array([pos.x, pos.y, heading]),
        'end': np.array([destination.x, destination.y, np.deg2rad(180)]),
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
    result_x = hybrid_astar_path.x_list
    result_y = hybrid_astar_path.y_list
    result_yaw = hybrid_astar_path.yaw_list

    import matplotlib.pyplot as plt
    from hybrid_a_star.car import plot_car
    plt.cla()
    plt.plot(ox, oy, ".k")
    plt.plot(result_x, result_y, "-r", label="Hybrid A* path")
    plt.grid(True)
    plt.axis("equal")
    c = 0
    for i_x, i_y, i_yaw in zip(result_x, result_y, hybrid_astar_path.yaw_list):
        if c % 5 == 0:
            plot_car(i_x, i_y, i_yaw)
        c += 1
    plt.savefig('test-2.png')
    # assert False

    # generate trajectory points
    trajectory = [TrajectoryPoint(Direction.FORWARD, x, y, MIN_SPEED, yaw) for x, y, yaw in zip(result_x, result_y, result_yaw)]
    trajectory[0].speed = vel.length()
    refine_trajectory(trajectory, direction, heading)

    # remove points that are too close
    new_trajectory = []
    for wp in trajectory:
        if pos.distance(wp) > WAYPOINT_THRESHOLD:
            new_trajectory.append(wp)
    
    return new_trajectory

# TODO: get data from actual GNSS sensor instead of getting
# perfect vehicle data from CARLA
class CarlaGnssSensor():
    def __init__(self, actor):
        self.actor = actor

    def get_location(self) -> list[float]:
        loc = self.actor.get_location()
        return [loc.x, loc.y]

    def get_velocity(self) -> list[float]:
        vel = self.actor.get_velocity()
        return [vel.x, vel.y]
    
    def get_heading(self):
        return np.deg2rad(self.actor.get_transform().rotation.yaw)

class CarlaCar():
    def __init__(self, world, blueprint, spawn_point, destination, debug=False):
        self.world = world
        self.actor = world.spawn_actor(blueprint, spawn_point)
        self.gnss_sensor = CarlaGnssSensor(self.actor)
        self.car = Car(self.actor.get_location(), self.actor.get_velocity(), destination, self.gnss_sensor)

        self.debug = debug
        if debug:
            self.debug_init(spawn_point, self.car.destination)

    def run_step(self):
        self.actor.apply_control(self.car.run_step())
        if self.debug:
            self.debug_step()

    def debug_init(self, spawn_point, destination):
        self.world.debug.draw_string(spawn_point.location, 'start', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
        self.world.debug.draw_string(destination, 'end', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

    def debug_step(self):
        self.world.debug.draw_string(self.car.pos, 'X', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=1.0, persistent_lines=True)
        for loc in self.car.trajectory:
            self.world.debug.draw_string(carla.Location(x=loc.x, y=loc.y), 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=1.0, persistent_lines=True)
        for pt in self.car.guidance_wps:
            self.world.debug.draw_string(carla.Location(x=pt[0], y=pt[1]), 'o', draw_shadow=False, color=carla.Color(r=255, g=255, b=0), life_time=1.0, persistent_lines=True)

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
        destination_bb = self.car.destination_bb
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
    def __init__(self, destination: list[float], gnss_sensor: CarlaGnssSensor):
        self.heading: float = 0
        self.cur = TrajectoryPoint(Direction.REVERSE, 0, 0, 0, 0)
        self.ps: float = 0
        self.obs: list[list[float]] = []
        self.destination = TrajectoryPoint(Direction.REVERSE, destination[0], destination[1], 0, 0)
        self.controller = VehiclePIDController({'K_P': 2, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}, {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': 0.05})
        self.gnss_sensor = gnss_sensor
        self.trajectory: list[TrajectoryPoint] = []
        self.mode = Mode.DRIVING
        self.direction = Direction.REVERSE

    def perceive(self):
        self.cur.x, self.cur.y = self.gnss_sensor.get_location()
        self.pos = self.gnss_sensor.get_location()
        self.vel = self.gnss_sensor.get_velocity()
        self.heading = self.gnss_sensor.get_heading()
        # TODO: get/update obstacles and lane waypoints from sensor data only,
        # adding data to graph as needed

    def plan(self):
        # TODO: handle planning for exploration phase
        # if we're at destination, stop
        # TODO: also stop if unexpected obstacle detected
        pos = self.pos
        destination = self.destination
        distance_to_destination = distance(pos, destination)
        if self.mode == Mode.PARKED or distance_to_destination < DESTINATION_THRESHOLD:
            self.mode = Mode.PARKED
            return STOP_CONTROL

        # switch to parking mode if close to destination
        if self.mode == Mode.DRIVING and distance_to_destination < 25:
            self.mode = Mode.PARKING

        # plan guidance wps if needed
        guidance_wps = self.guidance_wps
        lane_wps = self.lane_wps
        if not guidance_wps:
            # find closest waypoints to pos and destination
            pos_lane_wp = min(range(len(lane_wps)), key=lambda i: pos.distance(carla.Location(x=lane_wps[i][0], y=lane_wps[i][1])))
            destination_lane_wp = min(range(len(lane_wps)), key=lambda i: destination.distance(carla.Location(x=lane_wps[i][0], y=lane_wps[i][1])))

            # find shortest path
            G = nx.Graph()
            for i, waypoint in enumerate(lane_wps):
                G.add_node(i, pos=(waypoint[0], waypoint[1]))
            for i in range(len(lane_wps) - 1):
                G.add_edge(i, i + 1)
            guidance_wps.append([pos.x, pos.y])
            for wp in map(lambda i: lane_wps[i], nx.shortest_path(G, source=pos_lane_wp, target=destination_lane_wp)):
                guidance_wps.append(wp)

        # remove visited points from trajectory
        trajectory = self.trajectory
        num_to_remove = 0
        for loc in trajectory:
            if distance(pos, loc) < WAYPOINT_THRESHOLD:
                num_to_remove += 1
            else:
                break
        if num_to_remove > 0:
            trajectory = self.trajectory = trajectory[num_to_remove:]

        # replan trajectory if needed
        # if the trajectory is too small and doesn't end at the destination, replan (to extend)
        # if the trajectory is too far, replan (to fix)
        should_extend = len(trajectory) < REPLAN_THRESHOLD and (len(trajectory) == 0 or distance(destination, trajectory[-1]) > DESTINATION_THRESHOLD)
        should_fix = len(trajectory) > 0 and distance(pos, trajectory[0]) > 2
        if should_extend or should_fix:
            if self.mode == Mode.DRIVING:
                new_trajectory, new_ps = plan_fot(self.ps, pos, self.vel, guidance_wps, self.obs, self.direction, self.heading)
                if new_trajectory:
                    self.ps = new_ps
                    trajectory = self.trajectory = new_trajectory
            elif self.mode == Mode.PARKING:
                new_trajectory = plan_hybrid_astar(pos, self.vel, self.direction, self.heading, destination, self.obs)
                if new_trajectory:
                    trajectory = self.trajectory = new_trajectory
            plot_trajectory(trajectory)
        if not trajectory: self.mode = Mode.FAILED; return STOP_CONTROL

        # check if the next waypoint is behind us, in which case we need to reverse
        wp = trajectory[0]
        self.direction = wp.direction
        ctrl = self.slow_controller.run_step(
            self.vel,
            mps_to_kmph(wp.speed),
            carla.Transform(location=carla.Location(x=pos.x, y=pos.y), rotation=carla.Rotation(yaw=np.rad2deg(self.heading))),
            carla.Transform(location=carla.Location(x=wp.x, y=wp.y), rotation=carla.Rotation(yaw=np.rad2deg(wp.angle))),
            wp.direction == Direction.REVERSE
        )
        # if wp.direction == Direction.REVERSE:
        #     ctrl.steer = -ctrl.steer
        #     ctrl.reverse = True
        return ctrl

    def asdfasdf(self):
        if self.bruh > 0:
            self.bruh -= 1
            return STOP_CONTROL
        next_wp = trajectory[0]
        vel = self.vel
        next_wp_vec = np.array([next_wp.x - pos.x, next_wp.y - pos.y])
        next_wp_vec /= np.linalg.norm(next_wp_vec)
        forward_vec = np.array([np.cos(np.deg2rad(self.heading)), np.sin(np.deg2rad(self.heading))])
        if self.reversing and np.dot(next_wp_vec, forward_vec) > 0.5:
            print("FORWARDING")
            self.reversing = False
            self.bruh = 10
            return STOP_CONTROL
        if not self.reversing and np.dot(next_wp_vec, forward_vec) < -0.5:
            print("REVERSING")
            self.reversing = True
            self.bruh = 10
            return STOP_CONTROL
        if self.reversing:
            # offset pos to be rear axles
            # ctrl = self.reverse_controller.run_step(carla.Vector3D(x=vel.x, y=vel.y), target_speed, pos, trajectory[0])
            next_wp = carla.Location(x=next_wp.x - 0.5*forward_vec[0], y=next_wp.y - 0.5*forward_vec[1])
            ctrl = self.reverse_controller.run_step(carla.Vector3D(x=vel.x, y=vel.y), target_speed, pos, next_wp)
            ctrl.steer = -ctrl.steer
            print(ctrl.steer)
            ctrl.reverse = True
        else:
            next_wp = carla.Location(x=next_wp.x + 0.5*forward_vec[0], y=next_wp.y + 0.5*forward_vec[1])
            # ctrl = (self.slow_controller if mps_to_kmph(self.vel.length()) < SLOWDOWN_CONSTANT else self.fast_controller).run_step(self.vel, target_speed, pos, trajectory[0])
            ctrl = (self.slow_controller if mps_to_kmph(self.vel.length()) < SLOWDOWN_CONSTANT else self.fast_controller).run_step(self.vel, target_speed, pos, next_wp)

        return ctrl

    def run_step(self):
        self.perceive()
        return self.plan()