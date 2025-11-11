import pygame,random
from pygame.locals import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class World():

    def __init__(self, screen):
        self.screen = screen

        self.particles = []
        self.uuvs = []
        self.explosions = []

        self.controllable_uuv = None

        self.screen_width = self.screen.get_size()[0]
        self.screen_height = self.screen.get_size()[1]

        self.spawn_spacing = 10

        self.id_tracker = 0

        self.agent_rewards = {}  # {agent_id: reward}

        self.use_policy_after = 50

    def get_and_clear_reward(self, agent_id):
        """Get accumulated reward and reset"""
        reward = self.agent_rewards.get(agent_id, 0)
        self.agent_rewards[agent_id] = 0  # reset to default
        return reward

    def set_reward(self, agent_id, reward):
        """Accumulate rewards for an agent"""
        self.agent_rewards[agent_id] = reward

    def add_particle(self, x, y, radius, color):
        self.particles.append(Particle(x, y, radius, color, self, self.screen))

    def add_swarm_uuv_random(self, color, policy_net):
        x = random.randint(50, self.screen_width - 50)
        y = random.randint(50, self.screen_height - 50)
        dir = np.array([random.uniform(-1,1), random.uniform(-1,1)])
        unit_vec_dir = dir / np.linalg.norm(dir)

        self.add_swarm_uuv(x, y, unit_vec_dir, color, policy_net)

    def add_enemy_uuv_random(self, color, decision_making):
        enemy_x = random.randint(100, self.screen_width - 100)
        enemy_y = random.randint(100, self.screen_height - 100)

        for uuv in self.uuvs:
            if distance((enemy_x, enemy_y), (uuv.x, uuv.y)) < self.spawn_spacing:
                enemy_x = random.randint(100, self.screen_width - 100)
                enemy_y = random.randint(100, self.screen_height - 100)

        self.add_enemy_uuv(enemy_x, enemy_y, color, decision_making)

    def add_swarm_uuv(self, x, y, direction, color, policy_net):
        self.uuvs.append(SwarmUUV(x, y, direction, 5, color, policy_net, self, self.screen, self.id_tracker))
        self.id_tracker += 1

    def add_controllable_uuv(self, x, y, direction, color):
        _controllable_uuv = ControllableUUV(x, y, direction, 5, color, self, self.screen, self.id_tracker)
        self.uuvs.append(_controllable_uuv)
        self.controllable_uuv = _controllable_uuv
        self.id_tracker += 1

    def add_enemy_uuv(self, x, y, color, decision_making):
        self.uuvs.append(EnemyUUV(x, y, [0,1], 5, color, decision_making, self, self.screen, self.id_tracker))
        self.id_tracker += 1

    def add_explosion(self, x, y, duration, radius, color):
        self.explosions.append(Explosion(x, y, duration, radius, color, self, self.screen))

    def tick(self):
        for p in self.particles:
            p.tick()
        for u in self.uuvs:
            u.tick()
        for e in self.explosions:
            e.tick()

        self.screen_width = self.screen.get_size()[0]
        self.screen_height = self.screen.get_size()[1]

    def reset(self):
        self.uuvs = []
        self.particles = []
        self.explosions = []
        self.controllable_uuv = None
        self.agent_rewards = {}
        self.id_tracker = 0

class Particle:

    def __init__(self, startx, starty, radius, color, world:World, screen):
        self.world = world
        self.screen = screen

        self.x = startx
        self.y = starty
        self.radius = radius

        self.color = color

    def tick(self):
        pygame.draw.circle(self.screen, self.color, (self.x, self.y), self.radius)


class UUV(Particle):

    def __init__(self, startx, starty, direction, radius, color, world:World, screen, id):
        super().__init__(startx, starty, radius, color, world, screen)

        self.id = id

        self.direction = direction

        self.vel_vec = np.array([0, 0])

        self.acl_vec = np.array([0, 0])

        self.waypoints = []

        self.tick_counter = 0

    def tick(self):
        super().tick()
        self.tick_counter += 1
        pygame.draw.line(self.screen, self.color, (self.x, self.y), (self.x + 10 * self.direction[0], self.y + 10 * self.direction[1]), 1)

        # calculate physics
        self.acl_vec = np.add(self.acl_vec, -0.05 * self.vel_vec)  # friction
        self.vel_vec = np.add(self.vel_vec, self.acl_vec)  # apply acceleration
        self.vel_vec = np.add(random.uniform(-0.01, 0.01), self.vel_vec)  # randomness - current? idk

        # apply physics
        self.x = self.x + self.vel_vec[0]
        self.y = self.y + self.vel_vec[1]
        self.check_collision()

        # idk
        self.acl_vec = np.array([0,0])

    def increase_throttle(self, value=0.05):
        self.acl_vec = np.multiply(value, self.direction)

    def decrease_throttle(self, value=0.05):
        self.acl_vec = np.multiply(-1 * value, self.direction)

    def turn_left(self):
        current_angle = self.get_current_angle()
        new_angle = current_angle - 3
        self.set_angle_instantly(new_angle)

    def turn_right(self):
        current_angle = self.get_current_angle()
        new_angle = current_angle + 3
        self.set_angle_instantly(new_angle)

    def set_angle_instantly(self, angle):
        angle_rad = math.radians(angle)
        self.direction = (math.cos(angle_rad), math.sin(angle_rad))

    def check_collision(self):
        # should I be dead / no longer exist?
        for e in self.world.explosions:
            if (self.radius + e.radius) > distance((self.x, self.y), (e.x, e.y)):
                try_to_remove(self.world.uuvs, self)

        # wall collision y vector switch bc pygame down is positive
        if self.x > self.world.screen_width - 10:
            self.x = self.world.screen_width - 10
        if self.x < 10:
            self.x = 10
        if self.y > self.world.screen_height - 10:
            self.y = self.world.screen_height - 10
        if self.y < 10:
            self.y = 10

        # # if not a wall collision
        # if self.y < self.screen.get_size()[1] - 10 and self.y > 10 and self.x < self.screen.get_size()[0] - 10 and self.x > 10:
        for u in self.world.uuvs:
            if u.id != self.id:
                if (self.radius + u.radius) > distance((self.x, self.y), (u.x, u.y)):  # will need to change later when enemies aren't 1D
                    # explode
                    self.world.add_explosion(self.x, self.y, 50, 10, (255,200,0))

    def add_waypoint(self, waypoint):
        if waypoint[0] < 10:
            waypoint[0] = 10
        if waypoint[0] > self.world.screen_width - 10:
            waypoint[0] = self.world.screen_width - 10
        if waypoint[1] < 10:
            waypoint[1] = 10
        if waypoint[1] > self.world.screen_height - 10:
            waypoint[1] = self.world.screen_height - 10
        self.waypoints.append(waypoint)

    def go_to_waypoint(self):
        if len(self.waypoints) > 0:
            if distance((self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1])) < (self.radius):
                # arrived at waypoint
                self.waypoints.pop(0)
                self.vel_vec = np.array([0,0])
            else:
                # not yet arrived at waypoint
                pygame.draw.circle(self.screen, (150, 150, 150), (self.waypoints[0][0], self.waypoints[0][1]), 2 * self.radius / 3)
                pygame.draw.line(self.screen, (150, 150, 150), (self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1]), 2)

                current_angle = self.get_current_angle()
                desired_angle = math.atan2(self.waypoints[0][1] - self.y, self.waypoints[0][0] - self.x) * 180 / math.pi
                angle_diff = (desired_angle - current_angle + 180) % 360 - 180
                distance_to_waypoint = distance((self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1]))

                if abs(current_angle - desired_angle) < 3:  # if pointing generally the right direction
                    if np.linalg.norm(self.vel_vec) < 1:  # if not yet at max speed
                        self.increase_throttle(0.1)
                else:
                    if angle_diff > 0:
                        self.turn_right()
                    else:
                        self.turn_left()

    def get_current_angle(self):
        return math.atan2(self.direction[1], self.direction[0]) * 180 / math.pi

class SwarmUUV(UUV):

    def __init__(self, startx, starty, direction, radius, color, maddpg_agent, world: World, screen, id):
        super().__init__(startx, starty, direction, radius, color, world, screen, id)

        self.observations = []

        self.maddpg_agent = maddpg_agent

        self.current_action = None

        self.nearest_target = [0,0,0,0,0]

        self.observation_cone = 120

    def tick(self):
        self.world.set_reward(self.id, -0.05)

        # observing
        self.collect_observations()

        # decision making
        if self.tick_counter % self.world.use_policy_after == 0:
            self.take_action()
        self.go_to_waypoint()

        # physics, collision
        super().tick()

    def get_state(self):
        """Returns the current state vector"""
        smallest_dist = float("inf")
        closest_friendly = np.zeros(2 + len(self.vel_vec) + 1)
        for swarmuuv in self.world.uuvs:
            if isinstance(swarmuuv, SwarmUUV) and swarmuuv.id != self.id:
                tested_dist = distance((self.x, self.y), (swarmuuv.x, swarmuuv.y))
                if tested_dist < smallest_dist:
                    smallest_dist = tested_dist
                    closest_friendly = np.hstack([swarmuuv.x, swarmuuv.y, swarmuuv.vel_vec, 1])

        delta_friendly = [closest_friendly[0] - self.x, closest_friendly[1] - self.y, closest_friendly[2], closest_friendly[3], closest_friendly[4]]

        # np helps for speed here
        my_mesh_observations = []
        for swarmuuv in self.world.uuvs:
            if isinstance(swarmuuv, SwarmUUV) and len(swarmuuv.observations) > 0:
                my_mesh_observations.append(swarmuuv.observations)
        if len(my_mesh_observations) > 0:
            my_mesh_observations = np.concatenate(my_mesh_observations)
            # print(self.observations)
            # print(my_mesh_observations)
            # print()

        smallest_dist = float("inf")
        closest_enemy = np.zeros(2 + len(self.vel_vec) + 1)
        for observation in my_mesh_observations:
            tested_dist = distance((self.x, self.y), (observation[0], observation[1]))
            if tested_dist < smallest_dist:
                smallest_dist = tested_dist
                closest_enemy = [observation[0], observation[1], observation[2], observation[3], 1]  # x, y, vel_vel, true
                self.nearest_target = closest_enemy

        # limited memory of unseen enemies
        if closest_enemy[-1] == 0 or distance((self.x, self.y), (self.nearest_target[0], self.nearest_target[1])) < distance((self.x, self.y), (closest_enemy[0], closest_enemy[1])):  # calculated closest enemy doesn't exist OR is further than the most recent closest enemy
            delta_enemy = [self.nearest_target[0] - self.x, self.nearest_target[1] - self.y, self.nearest_target[2], self.nearest_target[3], self.nearest_target[4]]
        else:
            delta_enemy = [closest_enemy[0] - self.x, closest_enemy[1] - self.y, closest_enemy[2], closest_enemy[3], closest_enemy[4]]

        return np.hstack([delta_enemy, delta_friendly, self.direction, self.vel_vec, self.acl_vec]).astype(np.float32)

    def take_action(self):
        state = self.get_state()
        action = self.maddpg_agent.select_action(state)

        delta_x, delta_y = action

        self.waypoints = []
        self.add_waypoint([float(self.x + delta_x), float(self.y + delta_y)])
        self.current_action = action

    def collect_observations(self):
        self.observations = []
        for enemy in self.world.uuvs:
            if isinstance(enemy, EnemyUUV) or isinstance(enemy, ControllableUUV):
                current_angle = self.get_current_angle()
                angle_to_enemy = math.atan2(enemy.y - self.y, enemy.x - self.x) * 180 / math.pi
                angle_diff = abs(angle_to_enemy - current_angle)
                if angle_diff < self.observation_cone:
                    self.observations.append(np.hstack([enemy.x, enemy.y, enemy.vel_vec]))

    def check_collision(self):
        for u in self.world.uuvs:
            if u.id != self.id:
                if (self.radius + u.radius) > distance((self.x, self.y), (u.x, u.y)):
                    self.world.add_explosion(self.x, self.y, 50, 10, (255,200,0))
                    if isinstance(u, EnemyUUV):
                        self.world.set_reward(self.id, 10)
                    elif isinstance(u, SwarmUUV):
                        # self.world.set_reward(self.id, -0.5)
                        ...

        for e in self.world.explosions:
            if (self.radius + e.radius) > distance((self.x, self.y), (e.x, e.y)):
                try_to_remove(self.world.uuvs, self)

class EnemyUUV(UUV):

    def __init__(self, startx, starty, direction, radius, color, decision_making, world: World, screen, id):
        super().__init__(startx, starty, direction, radius, color, world, screen, id)
        self.decision_making = decision_making

    def tick(self):
        self.color = (100, 100, 10)
        smallest_dist = float("inf")
        closest_uuv = None
        for uuv in self.world.uuvs:

            if isinstance(uuv, SwarmUUV):  # should I render myself as observed
                swarmuuv_current_angle = uuv.get_current_angle()
                swarmuuv_angle_to_enemy = math.atan2(self.y - uuv.y, self.x - uuv.x) * 180 / math.pi
                angle_diff = abs(swarmuuv_angle_to_enemy - swarmuuv_current_angle)
                if angle_diff < uuv.observation_cone:
                    self.color = (255, 255, 50)

            if uuv.id != self.id:  # how I should run away (rudimentary)
                tested_dist = distance((self.x, self.y), (uuv.x, uuv.y))
                if tested_dist < smallest_dist:
                    smallest_dist = tested_dist
                    closest_uuv = uuv

        if self.decision_making == "static":
            ...
        elif self.decision_making == "random":
            if self.tick_counter % 250 == 0:
                self.waypoints = []
                x, y = random.randint(10, self.world.screen_width - 10), random.randint(10, self.world.screen_height - 10)
                self.waypoints.append((x, y))
        elif self.decision_making == "escape":
            if self.tick_counter % 10 == 0:
                self.waypoints = []
                if closest_uuv is not None:
                    dir_to_closest = np.array([closest_uuv.x - self.x, closest_uuv.y - self.y])
                    dir_to_closest = -25 * dir_to_closest / np.linalg.norm(dir_to_closest)
                    x, y = (dir_to_closest[0], dir_to_closest[1])
                    self.waypoints.append((self.x + x, self.y + y))

        self.go_to_waypoint()

        super().tick()

class ControllableUUV(UUV):

    def __init__(self, startx, starty, direction, radius, color, world: World, screen, id):
        super().__init__(startx, starty, direction, radius, color, world, screen, id)

    def tick(self):
        self.color = (200, 200, 200)
        for swarmuuv in self.world.uuvs:
            if isinstance(swarmuuv, SwarmUUV):
                swarmuuv_current_angle = swarmuuv.get_current_angle()
                swarmuuv_angle_to_enemy = math.atan2(self.y - swarmuuv.y, self.x - swarmuuv.x) * 180 / math.pi
                angle_diff = abs(swarmuuv_angle_to_enemy - swarmuuv_current_angle)
                if angle_diff < swarmuuv.observation_cone:
                    self.color = (255, 255, 255)
        super().tick()

class Explosion(Particle):

    def __init__(self, startx, starty, duration, radius, color, world:World, screen):
        super().__init__(startx, starty, radius, color, world, screen)
        self.lifetime = duration

    def tick(self):
        super().tick()
        self.lifetime -= 1
        if self.lifetime <=0 :
            try_to_remove(self.world.explosions, self)

def normalize_vector_l2(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def try_to_remove(list, object):
    try:
        list.remove(object)
    except ValueError:
        pass

def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
