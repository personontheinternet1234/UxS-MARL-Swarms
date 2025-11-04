import pygame,random
from pygame.locals import *
import numpy as np
import math

def normalize_vector_l2(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class World():

    def __init__(self, screen):
        self.screen = screen

        self.particles = []
        self.uuvs = []
        self.explosions = []

        self.controllable_uuv = None

    def add_particle(self, x, y, radius, color, id):
        self.particles.append(Particle(x, y, radius, color, self, self.screen))

    def add_vatn_uuv(self, x, y, direction, radius, color, id):
        self.uuvs.append(VatnUUV(x, y, direction, radius, color, self, self.screen, id))

    def add_controllable_uuv(self, x, y, direction, radius, color, id):
        _controllable_uuv = ControllableUUV(x, y, direction, radius, color, self, self.screen, id)
        self.uuvs.append(_controllable_uuv)
        self.controllable_uuv = _controllable_uuv

    def add_explosion(self, x, y, duration, radius, color):
        self.explosions.append(Explosion(x, y, duration, radius, color, self, self.screen))

    def tick(self):
        for p in self.particles:
            p.tick()
        for u in self.uuvs:
            u.tick()
        for e in self.explosions:
            e.tick()

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

    def tick(self):
        super().tick()
        pygame.draw.line(self.screen, self.color, (self.x, self.y), (self.x + 10 * self.direction[0], self.y + 10 * self.direction[1]), 1)

        # calculate physics
        self.acl_vec = np.add(self.acl_vec, -0.01 * self.vel_vec)  # friction
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
        current_angle = math.atan2(self.direction[1], self.direction[0]) * 180 / math.pi
        new_angle = current_angle - 3
        self.set_angle_instantly(new_angle)

    def turn_right(self):
        current_angle = math.atan2(self.direction[1], self.direction[0]) * 180 / math.pi
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
        if self.x > self.screen.get_size()[0] - 10:
            self.x = self.screen.get_size()[0] - 10
        if self.x < 10:
            self.x = 10
        if self.y > self.screen.get_size()[1] - 10:
            self.y = self.screen.get_size()[1] - 10
        if self.y < 10:
            self.y = 10

        # # if not a wall collision
        # if self.y < self.screen.get_size()[1] - 10 and self.y > 10 and self.x < self.screen.get_size()[0] - 10 and self.x > 10:
        for u in self.world.uuvs:
            if u.id != self.id:
                if (self.radius + u.radius) > distance((self.x, self.y), (u.x, u.y)):  # will need to change later when targets aren't 1D
                    # explode
                    self.world.add_explosion(self.x, self.y, 50, 10, (255,200,0))

class VatnUUV(UUV):

    def __init__(self, startx, starty, direction, radius, color, world: World, screen, id):
        super().__init__(startx, starty, direction, radius, color, world, screen, id)

        self.waypoints = []

        self.observations = []


    def tick(self):
        # observing
        self.collect_observations()

        # decision making

        self.go_to_waypoint()

        # physics
        super().tick()

    def take_action(self):
        ...

    def collect_observations(self):
        self.observations = []
        for enemy in self.world.uuvs:
            if isinstance(enemy, ControllableUUV):
                current_angle = math.atan2(self.direction[1], self.direction[0]) * 180 / math.pi
                angle_to_enemy = math.atan2(enemy.x - self.y, enemy.y - self.x) * 180 / math.pi
                angle_diff = abs(angle_to_enemy - current_angle)
                if angle_diff < 30:
                    self.observations.append((enemy.x, enemy.y))

    def add_waypoint(self, waypoint):
        self.waypoints.append(waypoint)

    def go_to_waypoint(self):
        if len(self.waypoints) > 0:
            if distance((self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1])) < (self.radius):
                # arrived at waypoint
                self.waypoints.pop(0)
                self.vel_vec = np.array([0,0])
            else:
                # not yet arrived at waypoint
                pygame.draw.line(self.screen, np.multiply(self.color, 0.5), (self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1]), 1)

                current_angle = math.atan2(self.direction[1], self.direction[0]) * 180 / math.pi
                desired_angle = math.atan2(self.waypoints[0][1] - self.y, self.waypoints[0][0] - self.x) * 180 / math.pi
                angle_diff = (desired_angle - current_angle + 180) % 360 - 180
                distance_to_waypoint = distance((self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1]))

                if abs(current_angle - desired_angle) < 3:  # if pointing generally the right direction
                    if np.linalg.norm(self.vel_vec) < 1:  # if not yet at max speed
                        self.increase_throttle()
                else:
                    if angle_diff > 0:
                        self.turn_right()
                    else:
                        self.turn_left()

class ControllableUUV(UUV):

    def __init__(self, startx, starty, direction, radius, color, world: World, screen, id):
        super().__init__(startx, starty, direction, radius, color, world, screen, id)

    def tick(self):
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

def try_to_remove(list, object):
    try:
        list.remove(object)
    except ValueError:
        pass

def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
