import pygame,random
from pygame.locals import *
import numpy as np
import math
from agents import *



pygame.init()
start_width = 500
start_height = 500
screen = pygame.display.set_mode((start_width, start_height), pygame.RESIZABLE)

black = (0,0,0)

white = (255, 255, 255)
blue = (50,50,160)
red = (160, 50, 50)
yellow = (160, 160, 50)
color_var = white

clock = pygame.time.Clock()

pygame.display.set_caption("Particle Sim")

particles = []
id = 0
exit_flag = False
my_world = World(screen)

while not exit_flag:
    keys = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            color = blue
            dir = np.array([random.uniform(-1,1),random.uniform(-1,1)])
            unit_vec_dir = dir / np.linalg.norm(dir)
            my_world.add_uuv(mouse_x, mouse_y, unit_vec_dir, 2, color, id)
            id += 1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                color = yellow
                my_world.add_controllable_uuv(mouse_x, mouse_y, np.array([0,1]), 2, color, id)
                id += 1
            if event.key == pygame.K_p:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                color = white
                my_world.add_particle(mouse_x, mouse_y, 3, color, id)
                for uuv in my_world.uuvs:
                    uuv.waypoints.append((mouse_x, mouse_y))
                id += 1
            if event.key == pygame.K_c:
                my_world.uuvs = []
                my_world.particles = []
                my_world.explosions = []
                my_world.controllable_uuv = None
                id = 0
        if event.type == pygame.VIDEORESIZE:
            width, height = event.w, event.h
            screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        if event.type == QUIT:
            exit_flag = True

    if my_world.controllable_uuv != None:
        if keys[pygame.K_w]:
            my_world.controllable_uuv.increase_throttle()
        if keys[pygame.K_s]:
            my_world.controllable_uuv.decrease_throttle()
        if keys[pygame.K_a]:
            my_world.controllable_uuv.turn_left()
        if keys[pygame.K_d]:
            my_world.controllable_uuv.turn_right()

    screen.fill(black)
    my_world.tick()
    pygame.display.flip()
    clock.tick(50)

pygame.quit()
