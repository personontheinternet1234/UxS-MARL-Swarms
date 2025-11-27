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

        self.use_policy_after = 0  # default ticks before policy used - tends to be changed
        self.personal_state_dim = 0
        self.observed_object_state_dim = 0
        self.observable_enemies = 0
        self.observable_friendlies = 0

        self.color_allocator = ColorAllocator()

    def get_and_clear_reward(self, agent_id):
        reward = self.agent_rewards.get(agent_id, 0)
        self.agent_rewards[agent_id] = 0  # reset to default
        return reward

    def add_reward(self, agent_id, reward):
        self.agent_rewards[agent_id] = self.agent_rewards.get(agent_id, 0.0) + reward

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
        self.particles.append(Particle(x, y, radius, color, self))

    def add_swarm_uuv(self, x, y, direction, color, policy_net):
        uuv = SwarmUUV(x, y, direction, 5, self.sonar_range_forward, self.max_hops, color, policy_net, self, self.id_tracker)
        self.uuvs.append(uuv)
        self.uuv_lookup[uuv.id] = uuv
        self.id_tracker += 1

    def add_controllable_uuv(self, x, y, direction, color):
        _controllable_uuv = ControllableUUV(x, y, direction, 5, color, self, self.screen, self.id_tracker)
        self.uuvs.append(_controllable_uuv)
        self.controllable_uuv = _controllable_uuv
        self.id_tracker += 1

    def add_enemy_uuv(self, x, y, color, decision_making):
        self.uuvs.append(EnemyUUV(x, y, [0,1], 7, color, decision_making, self, self.id_tracker))
        self.id_tracker += 1

    def add_explosion(self, x, y, duration, radius, color):
        self.explosions.append(Explosion(x, y, duration, radius, color, self))

    def add_barrier(self, x, y, direction, radius, color):
        self.barriers.append(Barrier(x, y, direction, radius, color, self))

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

    def __init__(self, startx, starty, radius, color, world:World):
        self.world = world
        self.screen = world.screen

        self.x = startx
        self.y = starty
        self.radius = radius

        self.color = color

    def tick(self):
        pygame.draw.circle(self.screen, self.color, (self.x, self.y), self.radius)


class UUV(Particle):

    def __init__(self, startx, starty, direction, radius, color, world:World, id):
        super().__init__(startx, starty, radius, color, world)

        self.id = id

        self.direction = direction

        self.vel_vec = np.array([0, 0])

        self.acl_vec = np.array([0, 0])

        self.max_speed = 1
        self.max_throttle = 0.1
        self.turn_rate = 3

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

    def turn_left(self, value):
        current_angle = self.get_current_angle()
        new_angle = current_angle - value
        self.set_angle_instantly(new_angle)

    def turn_right(self, value):
        current_angle = self.get_current_angle()
        new_angle = current_angle + value
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
                last_waypoint = (self.x, self.y)
                for waypoint in self.waypoints:
                    pygame.draw.circle(self.screen, self.waypoint_color, waypoint, 2 * self.radius / 3)
                    pygame.draw.line(self.screen, self.waypoint_color, last_waypoint, waypoint, 2)
                    last_waypoint = waypoint

                current_angle = self.get_current_angle()
                desired_angle = math.atan2(self.waypoints[0][1] - self.y, self.waypoints[0][0] - self.x) * 180 / math.pi
                angle_diff = (desired_angle - current_angle + 180) % 360 - 180
                distance_to_waypoint = distance((self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1]))

                if abs(angle_diff) < 3:  # if pointing generally the right direction
                    if np.linalg.norm(self.vel_vec) < self.max_speed:  # if not yet at max speed
                        self.increase_throttle(self.max_throttle)
                else:
                    if angle_diff > 0:
                        self.turn_right(self.turn_rate)
                    else:
                        self.turn_left(self.turn_rate)

    def get_current_angle(self):
        return math.atan2(self.direction[1], self.direction[0]) * 180 / math.pi

class SwarmUUV(UUV):

    def __init__(self, startx, starty, direction, radius, sonar_range_forward, max_hops, color, maddpg_agent:MADDPGAgent, world: World, id):
        super().__init__(startx, starty, direction, radius, color, world, id)

        self.max_speed = 2
        self.max_throttle = 0.4
        self.turn_rate = 3

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
        if self.tick_counter % self.world.use_policy_after == 0:
            self.world.add_reward(self.id, -0.1)

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

    def collect_observations(self):
        self.observations = []
        for enemy in self.world.uuvs:
            if isinstance(enemy, EnemyUUV) or isinstance(enemy, ControllableUUV):
                current_angle = self.get_current_angle()
                angle_to_enemy = math.atan2(enemy.y - self.y, enemy.x - self.x) * 180 / math.pi
                angle_diff = abs(angle_to_enemy - current_angle)
                if angle_diff < self.observation_cone and distance((self.x, self.y), (enemy.x, enemy.y)) <= self.sonar_range_forward:
                    self.observations.append(np.hstack([enemy.x, enemy.y, enemy.vel_vec]))

    def get_personal_state(self):
        return np.hstack([self.direction, self.vel_vec, self.acl_vec]).astype(np.float32)

    def get_observation_state(self):
        if not self.world.mesh_ans:
            full_obs_enemies = []
            for uuv in self.world.uuvs:
                if isinstance(uuv, SwarmUUV) and len(uuv.observations) > 0:
                    full_obs_enemies.append(uuv.observations)
            full_obs_enemies = np.concatenate(full_obs_enemies) if len(full_obs_enemies) > 0 else np.empty((0, self.world.observed_object_state_dim - 1))
        else:
            full_obs_enemies = np.asarray(self.observations, dtype=np.float32)
        full_obs_enemies = np.asarray(full_obs_enemies, dtype=np.float32)
        if full_obs_enemies.ndim == 1:
            full_obs_enemies = full_obs_enemies.reshape(1, -1)
        enemy_positions = full_obs_enemies[:, :2]
        deltas = enemy_positions - np.array([self.x, self.y])
        dists = np.linalg.norm(deltas, axis=1)
        idx = np.argsort(dists)[:self.world.observable_enemies]
        selected = full_obs_enemies[idx]
        selected_deltas = deltas[idx]
        n = selected_deltas.shape[0]
        enemies = np.zeros((self.world.observable_enemies, self.world.observed_object_state_dim), dtype=np.float32)
        enemies[:n, 0:2] = selected_deltas
        enemies[:n, 2:self.world.observed_object_state_dim - 1] = selected[:, 2:self.world.observed_object_state_dim - 1]
        enemies[:n, self.world.observed_object_state_dim - 1] = 1.0  # uuv actually here, not padded / otherwise 0

        if len(idx) > 0:
            smallest_dist = dists[idx[0]]
            if smallest_dist < self.last_distance:
                self.world.add_reward(self.id, min(0.2 * (self.last_distance - smallest_dist), 1.0))
                self.last_distance = smallest_dist

        if not self.world.mesh_ans:
            full_obs_friendlies = []
            for uuv in self.world.uuvs:
                if isinstance(uuv, SwarmUUV):
                    full_obs_friendlies.append([uuv.x, uuv.y, uuv.vel_vec[0], uuv.vel_vec[1]])
            full_obs_friendlies = np.concatenate(full_obs_friendlies) if len(full_obs_friendlies) > 0 else np.empty((0, self.world.observed_object_state_dim - 1))
        else:
            # TODO: realistic x,y,vel reporting via mesh packets, rn cheating a bit
            full_obs_friendlies = []
            for uuv in self.world.uuvs:
                if isinstance(uuv, SwarmUUV) and uuv.id in self.mesh:
                    full_obs_friendlies.append([uuv.x - self.x, uuv.y - self.y, uuv.vel_vec, 1])
            full_obs_friendlies = np.asarray(full_obs_friendlies, dtype=np.float32)
        full_obs_friendlies = np.asarray(full_obs_friendlies, dtype=np.float32)
        if full_obs_friendlies.ndim == 1:
            full_obs_friendlies = full_obs_friendlies.reshape(1, -1)
        friendly_positions = full_obs_friendlies[:, :2]
        deltas = friendly_positions - np.array([self.x, self.y])
        dists = np.linalg.norm(deltas, axis=1)
        idx = np.argsort(dists)[:self.world.observable_friendlies]
        selected = full_obs_friendlies[idx]
        selected_deltas = deltas[idx]
        n = selected_deltas.shape[0]
        friendlies = np.zeros((self.world.observable_friendlies, self.world.observed_object_state_dim), dtype=np.float32)
        friendlies[:n, 0:2] = selected_deltas
        friendlies[:n, 2:self.world.observed_object_state_dim - 1] = selected[:, 2:self.world.observed_object_state_dim - 1]
        friendlies[:n, self.world.observed_object_state_dim - 1] = 1.0  # uuv actually here, not padded / otherwise 0

        result = np.vstack((enemies, friendlies))
        return result

    def get_state(self):
        personal_state = self.get_personal_state()
        observation_state = self.get_observation_state().flatten()
        return np.concatenate([personal_state, observation_state])

    def take_action(self):
        state = self.get_state()
        exploring, action = self.maddpg_agent.select_action(state)
        self.waypoint_color = (100, 200, 100) if exploring else (150, 150, 150)
        delta_x, delta_y = action

        self.waypoints = []
        self.add_waypoint([float(self.x + delta_x), float(self.y + delta_y)])

        self.current_action = action

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
                        self.world.add_reward(self.id, 15)
                    elif isinstance(u, SwarmUUV):
                        self.world.add_reward(self.id, -1)

        for b in self.world.barriers:
            b_start_x = b.x - int(0.5 * (b.radius * b.direction[0]))
            b_start_y = b.y - int(0.5 * (b.radius * b.direction[1]))
            b_end_x = b.x + int(0.5 * (b.radius * b.direction[0]))
            b_end_y = b.y + int(0.5 * (b.radius * b.direction[1]))
            if dist_point_to_segment(self.x, self.y, b_start_x, b_start_y, b_end_x, b_end_y,) <= self.radius:
                self.world.add_explosion(self.x, self.y, 50, 10, (255,200,0))
                self.world.add_reward(self.id, -5)

        for e in self.world.explosions:
            if (self.radius + e.radius) > distance((self.x, self.y), (e.x, e.y)):
                try_to_remove(self.world.uuvs, self)

class EnemyUUV(UUV):

    def __init__(self, startx, starty, direction, radius, color, decision_making, world: World, id):
        super().__init__(startx, starty, direction, radius, color, world, id)
        self.decision_making = decision_making
        self.color = (200, 30, 30)
        self.waypoint_color = (190, 100, 100)

    def tick(self):
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
            if self.tick_counter % 100 == 0:
                self.waypoints = [self.waypoints[0]] if len(self.waypoints) > 0 else []
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
            pygame.draw.circle(self.screen, seen, (self.x, self.y), 1.3 * self.radius)
        super().tick()

    def go_to_waypoint(self):
        if len(self.waypoints) > 0:
            if distance((self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1])) < (self.radius):
                # arrived at waypoint
                # no stopping once here
                self.waypoints.pop(0)
            else:
                # not yet arrived at waypoint
                last_waypoint = (self.x, self.y)
                for waypoint in self.waypoints:
                    pygame.draw.circle(self.screen, self.waypoint_color, waypoint, 2 * self.radius / 3)
                    pygame.draw.line(self.screen, self.waypoint_color, last_waypoint, waypoint, 2)
                    last_waypoint = waypoint

                current_angle = self.get_current_angle()
                desired_angle = math.atan2(self.waypoints[0][1] - self.y, self.waypoints[0][0] - self.x) * 180 / math.pi
                angle_diff = (desired_angle - current_angle + 180) % 360 - 180
                distance_to_waypoint = distance((self.x, self.y), (self.waypoints[0][0], self.waypoints[0][1]))

                if np.linalg.norm(self.vel_vec) < self.max_speed:  # if not yet at max speed
                    self.increase_throttle(self.max_throttle)
                if angle_diff > 0:
                    self.turn_right(self.turn_rate)
                else:
                    self.turn_left(self.turn_rate)

class ControllableUUV(UUV):

    def __init__(self, startx, starty, direction, radius, color, world: World, screen, id):
        super().__init__(startx, starty, direction, radius, color, world, id)

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

    def __init__(self, startx, starty, duration, radius, color, world:World):
        super().__init__(startx, starty, radius, color, world)
        self.lifetime = duration

    def tick(self):
        super().tick()
        self.lifetime -= 1
        if self.lifetime <=0 :
            try_to_remove(self.world.explosions, self)

class Barrier(Particle):

    def __init__(self, startx, starty, direction, radius, color, world:World):
        super().__init__(startx, starty, radius, color, world)
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
    return ((px - bx)**2 + (py - by)**2)**0.5

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
