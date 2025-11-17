import pygame,random
from pygame.locals import *
import numpy as np
import math
import uuid
from rl_classes import *

class World():

    def __init__(self, screen):
        self.screen = screen

        self.particles = []
        self.uuvs = []
        self.explosions = []
        self.barriers = []
        self.controllable_uuv = None
        self.uuv_lookup = {}
        self.mesh_colors = {}  # {frozenset(mesh_ids): color}
        self.mesh_ans = False
        self.id_tracker = 0

        self.screen_width = self.screen.get_size()[0]
        self.screen_height = self.screen.get_size()[1]

        self.spawn_spacing = 10

        self.sonar_range_forward = 300
        self.sonar_range_heartbeat = 300
        self.max_hops = 2

        self.agent_rewards = {}  # {agent_id: reward}

        self.use_policy_after = 50  # default ticks before policy used - tends to be changed

        self.color_allocator = ColorAllocator()

    def get_and_clear_reward(self, agent_id):
        """Get accumulated reward and reset"""
        reward = self.agent_rewards.get(agent_id, 0)
        self.agent_rewards[agent_id] = 0  # reset to default
        return reward

    def set_reward(self, agent_id, reward):
        self.agent_rewards[agent_id] = reward

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

    def add_barrier_random(self, width, color):
        x = random.randint(50, self.screen_width - 50)
        y = random.randint(50, self.screen_height - 50)
        dir = np.array([random.uniform(-1,1), random.uniform(-1,1)])
        unit_vec_dir = dir / np.linalg.norm(dir)

        for uuv in self.uuvs:
            if distance((x, y), (uuv.x, uuv.y)) < self.spawn_spacing:
                x = random.randint(100, self.screen_width - 100)
                y = random.randint(100, self.screen_height - 100)

        self.add_barrier(x, y, unit_vec_dir, width, color)

    def add_particle(self, x, y, radius, color):
        self.particles.append(Particle(x, y, radius, color, self, self.screen))

    def add_swarm_uuv(self, x, y, direction, color, policy_net):
        uuv = SwarmUUV(x, y, direction, 5, self.sonar_range_forward, self.max_hops, color, policy_net, self, self.screen, self.id_tracker)
        self.uuvs.append(uuv)
        self.uuv_lookup[uuv.id] = uuv
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

    def add_barrier(self, x, y, direction, radius, color):
        self.barriers.append(Barrier(x, y, direction, radius, color, self, self.screen))

    def world_send_message_handling(self, asking_uuv, message):
        for u in self.uuvs:
            if isinstance(u, SwarmUUV):
                if distance((asking_uuv.x, asking_uuv.y), (u.x, u.y)) <= self.sonar_range_forward:
                    u.receive_message(message)

    def color_meshes_helper(self):
        if self.mesh_ans:
            for u in self.uuvs:
                if isinstance(u, SwarmUUV):
                    mesh_key = frozenset(u.mesh.keys())
                    if mesh_key not in self.mesh_colors:
                        self.mesh_colors[mesh_key] = self.color_allocator.next()

                    u.color = self.mesh_colors[mesh_key]

    def tick(self):
        for p in self.particles:
            p.tick()
        for u in self.uuvs:
            u.tick()
        for e in self.explosions:
            e.tick()
        for b in self.barriers:
            b.tick()

        for swarm_uuv in self.uuvs:
            if isinstance(swarm_uuv, SwarmUUV):
                for enemy_uuv in self.uuvs:
                    if isinstance(enemy_uuv, EnemyUUV):
                        dist = distance((swarm_uuv.x, swarm_uuv.y), (swarm_uuv.x, swarm_uuv.y))
                        if dist < swarm_uuv.last_distance:
                            swarm_uuv.last_distance = dist
                            self.set_reward(swarm_uuv.id, 2)

        self.uuv_lookup = {u.id: u for u in self.uuvs}

        self.color_meshes_helper()

        self.screen_width = self.screen.get_size()[0]
        self.screen_height = self.screen.get_size()[1]

    def reset(self):
        self.particles = []
        self.uuvs = []
        self.explosions = []
        self.barriers = []
        self.controllable_uuv = None
        self.uuv_lookup = {}
        self.mesh_colors = {}
        self.id_tracker = 0
        self.agent_rewards = {}

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

        self.waypoint_color = (150, 150, 150)

        self.tick_counter = 0

        self.spawn_prot = 0

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
        if self.tick_counter < self.spawn_prot:
            return

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
                pygame.draw.circle(self.screen, self.waypoint_color, (self.waypoints[0][0], self.waypoints[0][1]), 2 * self.radius / 3)
                pygame.draw.line(self.screen, self.waypoint_color, (self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1]), 2)

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

    def __init__(self, startx, starty, direction, radius, sonar_range_forward, max_hops, color, maddpg_agent: MADDPGAgent, world: World, screen, id):
        super().__init__(startx, starty, direction, radius, color, world, screen, id)

        self.mesh = {}

        self.sent_packets_already = []

        self.observations = []

        self.maddpg_agent = maddpg_agent

        self.current_action = None

        self.nearest_target = [0,0,0,0,0]

        self.last_distance = float("inf")

        self.observation_cone = 180

        self.sonar_range_forward = sonar_range_forward

        self.ttl = max_hops  # time to live / hops before message stops traveling

    def tick(self):
        self.world.set_reward(self.id, -0.05)

        # observing
        self.collect_observations()

        # meshing
        if self.tick_counter % self.world.use_policy_after == 0 and self.world.mesh_ans:
            if len(self.sent_packets_already) > 5:
                self.sent_packets_already.pop(0)
            self.send_heartbeat(self.ttl)
            self.update_personal_mesh()
            self.send_observations(self.ttl)

        # decision making
        if self.tick_counter % self.world.use_policy_after == 0:
            self.take_action()
        self.go_to_waypoint()

        # physics, collision
        super().tick()

    def take_action(self):
        state = self.get_state()
        enemy_feats = self.get_enemies()
        action, exploring = self.maddpg_agent.select_action(enemy_feats, state)

        self.waypoint_color = (150, 255, 150) if exploring else (150, 150, 150)

        delta_x = action[0]
        delta_y = action[1]

        self.waypoints = []
        self.add_waypoint([float(self.x + delta_x), float(self.y + delta_y)])
        self.current_action = action

    def get_state(self):
        """Returns the current state vector"""
        vel = self.vel_vec
        acl = self.acl_vec
        return np.hstack([self.direction, vel, acl]).astype(np.float32)

    def collect_observations(self):
        self.observations = []
        for enemy in self.world.uuvs:
            if isinstance(enemy, EnemyUUV) or isinstance(enemy, ControllableUUV):
                current_angle = self.get_current_angle()
                angle_to_enemy = math.atan2(enemy.y - self.y, enemy.x - self.x) * 180 / math.pi
                angle_diff = abs(angle_to_enemy - current_angle)
                if angle_diff < self.observation_cone and distance((self.x, self.y), (enemy.x, enemy.y)) <= self.sonar_range_forward:
                    feat = np.array([enemy.x, enemy.y, enemy.vel_vec[0], enemy.vel_vec[1]], dtype=np.float32)
                    self.observations.append(feat)

    def get_enemies(self):
        full_obs = []
        if not self.world.mesh_ans:
            for uuv in self.world.uuvs:
                if isinstance(uuv, SwarmUUV) and len(uuv.observations) > 0:
                    full_obs.append(uuv.observations)
            if len(full_obs) > 0:
                full_obs = np.concatenate(full_obs)
        else:
            full_obs = self.observations

        enemy_feats = []
        for enemy in full_obs:
            rel_x = enemy[0] - self.x
            rel_y = enemy[1] - self.y
            rx = float(rel_x)
            ry = float(rel_y)
            vx = float(enemy[2])
            vy = float(enemy[3])

            enemy_feat = [rx, ry, vx, vy]
            enemy_feats.append(enemy_feat)

        if len(enemy_feats) == 0:
            # no enemies → give a single zero enemy
            return np.zeros((1, 4), dtype=np.float32)
        return np.array(enemy_feats, dtype=np.float32)

    def send_message_to_world(self, message):
        self.world.world_send_message_handling(self, message)

    def pass_message_on(self, message):
        if message["ttl"] > 0:
            continued_message = message.copy()
            continued_message["ttl"] = int(message["ttl"]) - 1
            self.send_message_to_world(continued_message)  # passing it on

    def send_heartbeat(self, ttl):
        message = {"heartbeat":1, "ttl":ttl, "sender_id":self.id, "message_id":uuid.uuid4()}
        self.send_message_to_world(message)

    def send_observations(self, ttl):
        message = {"observations": self.observations, "ttl": ttl, "sender_id":self.id, "message_id":uuid.uuid4()}
        self.send_message_to_world(message)

    def receive_message(self, message):
        if message["message_id"] in self.sent_packets_already:  # to minimize redundant packet sending - if I've sent it before, why send it again
            return
        self.pass_message_on(message)
        self.sent_packets_already.append(message["message_id"])
        if message.get("heartbeat"):
            time_left = 5
            self.mesh[int(message["sender_id"])] = time_left
        if message.get("observations"):
            received_observations = message["observations"]
            for observation in received_observations:
                if not any(np.array_equal(observation, o) for o in self.observations):
                    self.observations.append(observation)

    def update_personal_mesh(self):
        for friendly_id in list(self.mesh.keys()):
            if self.mesh[friendly_id] == 0:
                del self.mesh[friendly_id]
            else:
                self.mesh[friendly_id] -= 1

    def check_collision(self):
        if self.tick_counter < self.spawn_prot:
            return

        for u in self.world.uuvs:
            if u.id != self.id:
                if (self.radius + u.radius) > distance((self.x, self.y), (u.x, u.y)):
                    self.world.add_explosion(self.x, self.y, 50, 10, (255,200,0))
                    if isinstance(u, EnemyUUV):
                        self.world.set_reward(self.id, 15)
                    elif isinstance(u, SwarmUUV):
                        # self.world.set_reward(self.id, -0.5)
                        ...

        for b in self.world.barriers:
            b_start_x = b.x - int(0.5 * (b.radius * b.direction[0]))
            b_start_y = b.y - int(0.5 * (b.radius * b.direction[1]))
            b_end_x = b.x + int(0.5 * (b.radius * b.direction[0]))
            b_end_y = b.y + int(0.5 * (b.radius * b.direction[1]))
            if dist_point_to_segment(self.x, self.y, b_start_x, b_start_y, b_end_x, b_end_y,) <= self.radius:
                self.world.add_explosion(self.x, self.y, 50, 10, (255,200,0))
                self.world.set_reward(self.id, -1.0)

        for e in self.world.explosions:
            if (self.radius + e.radius) > distance((self.x, self.y), (e.x, e.y)):
                try_to_remove(self.world.uuvs, self)

class EnemyUUV(UUV):

    def __init__(self, startx, starty, direction, radius, color, decision_making, world: World, screen, id):
        super().__init__(startx, starty, direction, radius, color, world, screen, id)
        self.decision_making = decision_making

    def tick(self):
        self.color = (190, 25, 25)
        seen = None

        smallest_dist = float("inf")
        closest_uuv = None
        for swarmuuv in self.world.uuvs:
            if isinstance(swarmuuv, SwarmUUV):  # should I render myself as observed
                swarmuuv_current_angle = swarmuuv.get_current_angle()
                swarmuuv_angle_to_enemy = math.atan2(self.y - swarmuuv.y, self.x - swarmuuv.x) * 180 / math.pi
                angle_diff = abs(swarmuuv_angle_to_enemy - swarmuuv_current_angle)
                if angle_diff < swarmuuv.observation_cone and distance((swarmuuv.x, swarmuuv.y), (self.x, self.y)) <= swarmuuv.sonar_range_forward:
                    seen = swarmuuv.color
            if swarmuuv.id != self.id:  # how I should run away (rudimentary)
                tested_dist = distance((self.x, self.y), (swarmuuv.x, swarmuuv.y))
                if tested_dist < smallest_dist:
                    smallest_dist = tested_dist
                    closest_uuv = swarmuuv

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

        if seen is not None:
            pygame.draw.circle(self.screen, seen, (self.x, self.y), 1.4 * self.radius)
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
                if angle_diff < swarmuuv.observation_cone and distance((swarmuuv.x, swarmuuv.y), (self.x, self.y)) <= swarmuuv.sonar_range_forward:
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

class Barrier(Particle):

    def __init__(self, startx, starty, direction, radius, color, world:World, screen):
        super().__init__(startx, starty, radius, color, world, screen)
        self.direction = direction

    def tick(self):
        start_x = self.x - int(0.5 * (self.radius * self.direction[0]))
        start_y = self.y - int(0.5 * (self.radius * self.direction[1]))
        end_x = self.x + int(0.5 * (self.radius * self.direction[0]))
        end_y = self.y + int(0.5 * (self.radius * self.direction[1]))
        pygame.draw.line(self.screen, self.color, (start_x, start_y), (end_x, end_y), 3)

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

def dist_point_to_segment(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx * wx + vy * wy

    if c1 <= 0:
        return (px - x1)**2 + (py - y1)**2

    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return (px - x2)**2 + (py - y2)**2

    b = c1 / c2
    bx = x1 + b * vx
    by = y1 + b * vy
    return (px - bx)**2 + (py - by)**2

class ColorAllocator:
    def __init__(self):
        self.palette = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
        (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
        (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
        (255, 255, 255), (100, 149, 237), (255, 140, 0),
        (34, 139, 34), (186, 85, 211), (72, 61, 139),
        (218, 165, 32), (127, 255, 212), (199, 21, 133), (30, 144, 255),
        (154, 205, 50), (255, 105, 180), (139, 0, 139), (60, 179, 113),
        (233, 150, 122), (0, 206, 209), (123, 104, 238)
        ]
        self.index = 0

    def next(self):
        if self.index < len(self.palette):
            c = self.palette[self.index]
            self.index += 1
            return c
        else:
            # fallback if exhausted
            import random
            return (random.randint(0,255),
                    random.randint(0,255),
                    random.randint(0,255))
