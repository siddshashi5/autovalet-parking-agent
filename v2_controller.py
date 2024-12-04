# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np
import carla

def get_speed(vel):
    """
    Compute speed of a vehicle in Km/h.

        :param vel: velocity
        :return: speed as a float in Km/h
    """
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

class VehiclePIDController:
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """


    def __init__(self, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3,
                 max_steering=1.0):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self.past_steering = 0.0
        self._lon_controller = PIDLongitudinalController(**args_longitudinal)
        self._lat_controller = PIDLateralController(offset, **args_lateral)

    def run_step(self, cur_speed, target_speed, cur, wp, is_reverse):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """

        acceleration = self._lon_controller.run_step(cur_speed, target_speed)
        current_steering = self._lat_controller.run_step(cur, wp, is_reverse)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        # if current_steering > self.past_steering + 0.1:
        #     current_steering = self.past_steering + 0.1
        # elif current_steering < self.past_steering - 0.1:
        #     current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering
        if is_reverse:
            control.reverse = True

        return control


    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_PID(self, args_lateral):
        """Changes the parameters of the PIDLateralController"""
        self._lat_controller.change_parameters(**args_lateral)

    def set_offset(self, offset):
        """Changes the offset"""
        self._lat_controller.set_offset(offset)


class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, current_speed, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, offset=0, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, cur, wp, is_reverse):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoint.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(cur, wp, is_reverse)

    def set_offset(self, offset):
        """Changes the offset"""
        self._offset = offset

    def _pid_control(self, cur, wp, is_reverse):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # front_cur = cur.offset(1)
        # front_wp = wp.offset(1)
        # back_cur = cur.offset(-1)
        # back_wp = wp.offset(-1)
        # forward_vec = np.array([np.cos(cur.angle), np.sin(cur.angle), 0])

        # front_wp_vec = np.array([front_wp.x - front_cur.x, front_wp.y - front_cur.y, 0])
        # front_dot = np.dot(forward_vec, front_wp_vec) / (np.linalg.norm(forward_vec) * np.linalg.norm(front_wp_vec))
        # front_cross = np.cross(forward_vec, front_wp_vec)
        # front_angle = np.acos(front_dot)
        # if front_cross[2] < 0:
        #     front_angle *= -1

        # back_wp_vec = np.array([back_wp.x - back_cur.x, back_wp.y - back_cur.y, 0])
        # back_dot = np.dot(forward_vec, back_wp_vec) / (np.linalg.norm(forward_vec) * np.linalg.norm(back_wp_vec))
        # back_cross = np.cross(forward_vec, back_wp_vec)
        # back_angle = np.acos(back_dot)
        # if back_cross[2] < 0:
        #     back_angle *= -1
        # if front_cross[2] < 0 and back_cross[2] > 0 or front_cross[2] > 0 and back_cross[2] < 0:
        #     back_angle *= -1

        # print(f'{np.rad2deg(front_angle):.2f} {np.rad2deg(back_angle):.2f}')
        # delta = 0.3*front_angle + 0.7*back_angle
        # delta *= 0.1
        # delta *= 2/(cur.speed*cur.speed) + 0.5

        # back_cur = cur.offset(-1) if not is_reverse else cur.offset(1)
        # back_wp = wp.offset(0) if not is_reverse else wp.offset(1)

        # alpha = np.atan2(back_wp.y - back_cur.y, back_wp.x - back_cur.x) - cur.angle
        # delta = np.atan2(2.0 * 3.0 * np.sin(alpha), np.sqrt((back_wp.x - back_cur.x) ** 2 + (back_wp.y - back_cur.y) ** 2))

        alpha = np.atan2(wp.y - cur.y, wp.x - cur.x) - cur.angle
        delta = np.atan2(2.0 * 3.0 * np.sin(alpha), np.sqrt((wp.x - cur.x) ** 2 + (wp.y - cur.y) ** 2))
        # delta = np.atan2(3.0 * (wp.angle - cur.angle), np.sqrt((back_wp.x - back_cur.x) ** 2 + (back_wp.y - back_cur.y) ** 2))
        
        # forward_vec = np.array([np.cos(cur.angle), np.sin(cur.angle), 0])
        # wp_vec = np.array([wp.x - cur.x, wp.y - cur.y, 0])
        # dot = np.dot(forward_vec, wp_vec) / (np.linalg.norm(forward_vec) * np.linalg.norm(wp_vec))
        # cross = np.cross(forward_vec, wp_vec)
        # delta = np.acos(dot)
        # if cross[2] < 0:
        #     delta *= -1
        
        # self._e_buffer.append(delta)
        # if len(self._e_buffer) >= 2:
        #     _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
        #     _ie = sum(self._e_buffer) * self._dt
        # else:
        #     _de = 0.0
        #     _ie = 0.0
        # k_p, k_d, k_i = 1, 0.1, 0.0
        # delta = (k_p * delta) + (k_d * _de) + (k_i * _ie)

        # dangle = (wp.angle - prev_wp.angle) / 0.5
        # delta = np.atan2(2.875/2 * dangle, 1)
        print(f'{np.rad2deg(cur.angle):.2f} {np.rad2deg(wp.angle):.2f} {np.rad2deg(delta):.2f}')

        # delta *= 1/(cur.speed*cur.speed) + 2

        # delta = (1.5/cur.speed) * (wp.angle - cur.angle)
        # delta = 0.2*np.atan2(3 * (wp.angle - cur.angle), cur.speed * 2 * cur.distance(wp) / (cur.speed + wp.speed))
        # delta = 1 * (wp.angle - cur.angle) / (cur.distance(wp) / cur.speed)
        # print(f'{np.rad2deg(cur.angle):.2f} {np.rad2deg(wp.angle):.2f} {np.rad2deg(delta):.2f}')
        # front_cur = cur.offset(1)
        # front_wp = wp.offset(1)
        # phi = wp.angle - cur.angle
        # e = np.atan2(0.1 * np.sqrt((front_wp.x - front_cur.x) ** 2 + (front_wp.y - front_cur.y) ** 2), cur.speed) 
        # print(f'{np.rad2deg(phi):.2f} {np.rad2deg(e):.2f}')
        # print()
        # phi = np.arctan2(np.sin(phi), np.cos(phi))
        # delta = phi + e

        # delta = wp.angle - cur.angle
        max_angle = np.deg2rad(70)
        delta = np.clip(delta, -max_angle, max_angle)
        delta /= max_angle

        # front_cur = cur.offset(1)
        # front_wp = wp.offset(1)
        # # heading_err = wp.angle - cur.angle
        # heading_err = wp.angle - cur.angle
        # cte = np.atan2(0.1 * np.sqrt((front_wp.x - front_cur.x) ** 2 + (front_wp.y - front_cur.y) ** 2), cur.speed)
        # delta = heading_err + cte
        # delta = np.arctan2(np.sin(delta), np.cos(delta))
        # delta = np.clip(delta, -np.pi / 6, np.pi / 6)
        # delta /= np.pi / 6

        return delta

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt