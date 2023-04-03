import airsim
import numpy as np
import math
import torch
import datetime
import time
import random

from airsim import MultirotorClient
from math import *
from airsim import YawMode


class f:
    # f.flag == 1 representative that UAV is in avoidance state
    flag = 0
    # When f.draw == 1, it will draw the flight path of the UAV
    draw = 0


class Multirotor:
    def __init__(self, sensor, human):
        # Which sensor will be used
        self.sensor = sensor
        # Whether in human-in-the-loop or not
        self.human = human
        # connect to the AirSim
        self.client = airsim.MultirotorClient()
        # Reset the AirSim environment
        self.client.reset()

        # Set the name of UAV
        name = "UAV1"
        self.client.enableApiControl(True, name)
        self.client.armDisarm(True, name)
        # UAV take-off
        self.client.takeoffAsync()

        # reset the avoidance state
        self.flag = 0

        # randomly generate target points distributed on the square boundary
        # four directions of east, west, south and north
        side_list = [1, 2, 3, 4]
        side = random.choice(side_list)
        if side == 1:
            self.tx = random.randint(-320, -280)
            self.ty = random.randint(-360, 270)
        elif side == 2:
            self.tx = random.randint(-300, 370)
            self.ty = random.randint(-330, -290)
        elif side == 3:
            self.tx = random.randint(350, 390)
            self.ty = random.randint(-360, 310)
        else:
            self.tx = random.randint(-300, 370)
            self.ty = random.randint(-380, -340)
        self.tz = -10

        # the UAV takes off to the height of 10 meters at a speed of 5 meters per second
        self.client.moveToZAsync(-10, 5, vehicle_name=name).join()
        # obtain UAV dynamics information: position and velocity
        self.ux, self.uy, self.uz, self.vx, self.vy, self.vz = self.get_kinematic_state()
        # set regional boundaries
        self.bound_x = [-400, 400]
        self.bound_y = [-400, 400]
        self.bound_z = [-60, 10]
        # set the safe distance
        self.d_safe = 15
        # initialize the distance between the UAV and the target point
        self.init_distance = self.get_distance()

    '''
    Obtain UAV kinematics information:position and velocity
    '''
    def get_kinematic_state(self):
        name = "UAV1"
        kinematic_state = self.client.simGetGroundTruthKinematics(vehicle_name=name)
        # position
        ux = float(kinematic_state.position.x_val)
        uy = float(kinematic_state.position.y_val)
        uz = float(kinematic_state.position.z_val)
        # velocity
        vx = float(kinematic_state.linear_velocity.x_val)
        vy = float(kinematic_state.linear_velocity.y_val)
        vz = float(kinematic_state.linear_velocity.z_val)

        return ux, uy, uz, vx, vy, vz

    '''
    Obtain the heading angle deviation angle of the UAV
    '''
    def get_deflection_angle(self):
        ux, uy, uz, vx, vy, vz = self.get_kinematic_state()
        # self.tx, self.ty, self.tz are the positions of the target point
        model_a = pow((self.tx - ux) ** 2 + (self.ty - uy) ** 2 + (self.tz - uz) ** 2, 0.5)
        model_b = pow(vx ** 2 + vy ** 2 + vz ** 2, 0.5)
        cos_ab = ((self.tx - ux) * vx + (self.ty - uy) * vy + (self.tz - uz) * vz) / (model_a * model_b)
        radius = acos(cos_ab)
        angle = np.rad2deg(radius)

        return angle

    '''
    Obtain the distance between the UAV and the target point
    '''
    def get_distance(self):
        ux, uy, uz, _, _, _ = self.get_kinematic_state()
        distance = pow((self.tx - ux) ** 2 + (self.ty - uy) ** 2 + (self.tz - uz) ** 2, 0.5)
        return distance

    '''
    Obtain UAV distance sensors data
    '''
    def get_distance_sensors_data(self, sensor):
        yaw_axis = 'A'
        # 12 distance sensors
        pitch_axis = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        prefix = "Distance"
        name = "UAV1"
        if sensor == 0:
            data = []
            for i in pitch_axis:
                dsn = prefix + yaw_axis + i
                distance = self.client.getDistanceSensorData(distance_sensor_name=dsn, vehicle_name=name).distance
                # The detection range of the sensor is set to 30 meters.
                if distance > 30:
                    distance = 30
                data.append(distance)
            return data
        else:
            dsn = "DistanceA1"
            distance = self.client.getDistanceSensorData(distance_sensor_name=dsn, vehicle_name=name).distance
            return distance

    '''
    Obtain UAV depth camera data
    '''
    def get_depth_image(self):
        thresh = 20
        # The pixels of the picture can be changed arbitrarily in the AirSim
        image_shape = (52, 52, 1)
        depth_image_request = airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)
        responses = self.client.simGetImages([depth_image_request], vehicle_name="UAV1")
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = depth_image.reshape(responses[0].height, responses[0].width)
        depth_image[depth_image > thresh] = thresh
        if len(depth_image) == 0:
            depth_image = np.zeros(image_shape)
        # The image is input into the pre-training model to obtain features with a dimension of 1 × 40.
        depth_model = torch.load('.../pre_model/depth_model/checkpoint.pk')
        depth_feature = depth_model(depth_image)

        return depth_feature

    '''
    Obtain UAV state
    '''
    def get_state(self):
        ux, uy, uz, vx, vy, vz = self.get_kinematic_state()
        if self.sensor == 0:
            position = np.array([self.tx - ux, self.ty - uy, self.tz - uz]) / 400
            target = np.array([self.get_distance() / self.init_distance])
            velocity = np.array([vx, vy, vz]) / 10
            angle = np.array([self.get_deflection_angle() / 180])
            sensor_data = np.array(self.get_distance_sensors_data(self.sensor)) / 20
            # state: 1 × 20
            state = np.append(position, target)    # 1 × 4
            state = np.append(state, velocity)     # 1 × 3
            state = np.append(state, angle)        # 1 × 1
            state = np.append(state, sensor_data)  # 1 × 12

            return state
        else:
            position = np.array([self.tx - ux, self.ty - uy, self.tz - uz]) / 400
            target = np.array([self.get_distance() / self.init_distance])
            velocity = np.array([vx, vy, vz]) / 10
            angle = np.array([self.get_deflection_angle() / 180])
            sensor_data = np.array(self.get_distance_sensors_data(self.sensor)) / 20
            depth_feature = np.array(self.get_depth_image())
            # state: 1 × 49
            state = np.append(position, target)      # 1 × 4
            state = np.append(state, velocity)       # 1 × 3
            state = np.append(state, angle)          # 1 × 1
            state = np.append(state, sensor_data)    # 1 × 1
            state = np.append(state, depth_feature)  # 1 × 40
            # state: 1 × 50
            if self.human == 1:
                human_model = torch.load('.../pre_model/human_model/checkpoint.pk')
                flag = human_model(state)            # flag is 0 or 1
                f.flag = flag
                state = np.append(state, flag)       # 1 × 1

            return state

    '''
    Step
    '''
    def step(self, action):
        done = self.if_done()
        second_reward = self.second_reward()
        arrive_reward = self.arrive_reward()
        yaw_reward = self.yaw_reward()
        z_distance_reward = self.z_distance_reward()
        collision_reward = 0
        step_reward = self.step_reward()
        cross_border_reward = self.cross_border_reward()

        # Judging whether the UAV collides
        name = "UAV1"
        if self.client.simGetCollisionInfo(vehicle_name=name).has_collided:
            collision_reward = collision_reward - 12
            done = True

        reward = arrive_reward + yaw_reward + collision_reward + step_reward + cross_border_reward \
                 + uav_distance_reward + z_distance_reward + second_reward

        # Set UAV flight mode
        yaw_mode = airsim.YawMode(False, 0)
        drivetrain = airsim.DrivetrainType.ForwardOnly

        # Actor network outputs three-direction acceleration
        ax = action[0]
        ay = action[1]
        az = action[2]

        # Obtain current kinematics information of UAV
        ux, uy, uz, vx, vy, vz = self.get_kinematic_state()
        self.client.moveByVelocityAsync(vx=vx + ax,
                                        vy=vy + ay,
                                        vz=vz + az,
                                        duration=0.5,
                                        drivetrain=drivetrain,
                                        yaw_mode=yaw_mode,
                                        vehicle_name=name)

        state_ = self.get_state()

        # Drawing UAV flight trajectory
        if f.draw == 1:

            pos = np.array([[ux], [uy], [uz]])
            point_reserve = [airsim.Vector3r(self.pos_reserve[0, 0], self.pos_reserve[1, 0], self.pos_reserve[2, 0]), ]
            point = [airsim.Vector3r(pos[0, 0], pos[1, 0], pos[2, 0])]
            self.client.simPlotLineStrip(
                point_reserve + point,
                color_rgba=[1.0, 0.0, 0.0, 1.0],
                thickness=50.0,
                duration=-1.0,
                is_persistent=True
            )
            self.pos_reserve = pos

        return state_, reward, done

    def if_done(self):
        ux, uy, uz, _, _, _ = self.get_kinematic_state()
        distance = self.get_distance()
        if distance < 30:
            return True

        if ux < self.bound_x[0] or ux > self.bound_x[1]:
            return True
        elif uy < self.bound_y[0] or uy > self.bound_y[1]:
            return True
        elif uz < self.bound_z[0] or uz > self.bound_z[1]:
            return True

        return False

    '''
    Flight altitude reward
    '''
    def z_distance_reward(self):
        _, _, uz, _, _, _ = self.get_kinematic_state()
        reward = 0
        z_distance = abs(uz[i] - self.tz)
        if z_distance > 20:
            reward += -0.02
        elif 20 >= z_distance > 10:
            reward += -0.015
        elif 10 >= z_distance > 5:
            reward += -0.01
        else:
            reward += 0.01

        return reward

    '''
    Cross the border reward
    '''
    def cross_border_reward(self):
        ux, uy, uz, _, _, _ = self.get_kinematic_state()
        reward = 0
        # if UAV flew out of the boundary
        if ux < self.bound_x[0] or ux > self.bound_x[1] or \
                uy < self.bound_y[0] or uy > self.bound_y[1] or \
                uz < self.bound_z[0] or uz > self.bound_z[1]:
            reward -= 12

        return reward

    '''
    Arrive reward
    '''
    def arrive_reward(self):
        ux, uy, uz, _, _, _ = self.get_kinematic_state()
        reward = 0
        model_a = pow((self.tx - ux) ** 2 + (self.ty - uy) ** 2 + (self.tz - uz) ** 2, 0.5)
        # The UAV arrives near the target point
        if model_a <= 30.0:
            reward += 12

        return reward

    '''
    Yaw reward
    '''
    def yaw_reward(self):
        # if UAV is not in avoidance state
        if f.flag == 0:
            yaw = self.get_deflection_angle()
            return -0.4 * (yaw / 180)
        # if UAV in avoidance state
        else:
            return 0

    '''
    Step reward
    '''
    def step_reward(self):
        reward = 0
        distance = self.get_distance()
        if distance < 100:
            distance = 100
        reward += -0.02 * 100 / distance

        return reward

    '''
    Second target reward
    '''
    def second_reward(self):
        reward = 0
        if f.flag == 1:
            yaw = self.avoidance()
            if abs(yaw) < 10:
                reward = 0.15
            elif 10 <= abs(yaw) < 20:
                reward = 0.1
            elif 20 <= abs(yaw) < 30 :
                reward = 0.05
            else:
                reward = 0
            return reward
        else:
            return reward

    '''
    When UAV in avoidance, calculate secondary target point
    '''
    def avoidance(self):
        yaw = self.get_deflection_angle()
        pitch_axis = ['1', '2', '3', '4', '5', '6', '7']
        data = []
        prefix = "Distance"
        yaw_list = [-90, -60, -30, 0, 30, 60, 90]
        yaw_min = 60
        for i in pitch_axis:
            dsn = prefix + 'C' + i
            distance = self.client.getDistanceSensorData(distance_sensor_name=dsn, vehicle_name='UAV1').distance
            if distance > 30:
                distance = 30
            data.append(distance)

        for i in range(6):
            if data[i] == 30:
                yaw_this = abs(yaw_list[i] - yaw)
                if yaw_this < yaw_min:
                    yaw_min = yaw_this

        return yaw_min


if __name__ == '__main__':
    print("")
