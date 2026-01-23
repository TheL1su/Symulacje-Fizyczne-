import pygame
import sys
import random
import numpy as np
from pygame.locals import *
pygame.init()


WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 700
FPS = 60
SIM_MIN_WIDTH = 20.0
cScale = min(WINDOW_WIDTH, WINDOW_HEIGHT) / SIM_MIN_WIDTH
simWidth = WINDOW_WIDTH / cScale
simHeight = WINDOW_HEIGHT / cScale
timeStep = 1/600.0
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Ball on Wire simulation")
clock = pygame.time.Clock()
dt = 1.0 / 60.0

def cX(x):
    return int(x * cScale)

def cY(y):
    return int(WINDOW_HEIGHT - y * cScale)

class Ball:

    def __init__(self,radius,pos,vel,colour):
        self.radius = radius
        self.pos = pos
        self.prev_pos = Vector2()
        self.vel = vel
        self.col = colour
        self.mass = np.pi * (radius ** 2)
    
    def draw_circle(self):
        pygame.draw.circle(screen,self.col,
        (cX(self.pos.as_tuple()[0]), cY(self.pos.as_tuple()[1])),
        int(cScale * self.radius))

    def start_step(self,dt):
        self.vel.add(gravity, dt)
        self.prev_pos = self.pos.clone()
        self.pos.add(self.vel, dt)

    def keep_on_wire(self, center, radius):
        dir = Vector2()
        dir.subtract_vectors(self.pos, center)
        length = dir.length()
        if length == 0.0:
            return 0.0
        dir.scale(1.0 / length)
        lam = radius - length
        self.pos.add(dir, lam)

    def end_step(self,dt):
        self.vel.subtract_vectors(self.pos, self.prev_pos)
        self.vel.scale(1.0 / dt)

def handleBeadBeadCollision(bead1, bead2):
    restitution = 1.0
    dir = Vector2()
    dir.subtract_vectors(bead2.pos, bead1.pos)
    d = dir.length()
    if (d == 0.0 or d > bead1.radius + bead2.radius):
        return
    dir.scale(1.0 / d)
    corr = (bead1.radius + bead2.radius - d) / 2.0
    bead1.pos.add(dir, -corr)
    bead2.pos.add(dir, corr)

    v1 = bead1.vel.dot(dir)
    v2 = bead2.vel.dot(dir)

    m1 = bead1.mass
    m2 = bead2.mass
    newV1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2) * restitution) / (m1 + m2)
    newV2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1) * restitution) / (m1 + m2)

    bead1.vel.add(dir, newV1 - v1)
    bead2.vel.add(dir, newV2 - v2)
	

class Vector2:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def clone(self):
        return Vector2(self.x, self.y)

    def add(self, v, s=1.0):
        self.x += v.x * s
        self.y += v.y * s
        return self

    def subtract(self, v, s=1.0):
        self.x -= v.x * s
        self.y -= v.y * s
        return self

    def subtract_vectors(self, a, b):
        self.x = a.x - b.x
        self.y = a.y - b.y
        return self

    def length(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def scale(self, scale):
        self.x *= scale
        self.y *= scale
        return self

    def as_tuple(self):
        return (self.x, self.y)
    def dot(self,v):
        return self.x * v.x + self.y * v.y
    

def Run(simWidth,FPS):
    
    dist_from_sun = 8
    set_of_balls = []
    num_of_balls = 10
    for i in range(0,num_of_balls):
        interval = [i*(2*np.pi/num_of_balls),(i+1)*(2*np.pi/num_of_balls)*0.85]
        radius = random.uniform(0.2,1)
        alpha = random.uniform(interval[0],interval[1])
        pos = Vector2(dist_from_sun*np.cos(alpha) + simWidth/2,dist_from_sun*np.sin(alpha) + simHeight/2)
        vel = Vector2()
        colour = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
        set_of_balls.append(Ball(radius,pos,vel,colour))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        pygame.draw.circle(screen,(0,0,0),
        (cX(simWidth/2), cY(simHeight/2)),
        int(cScale * dist_from_sun),width = 1)

        for i in range(len(set_of_balls)):
            set_of_balls[i].draw_circle()
            set_of_balls[i].start_step(dt)
            set_of_balls[i].keep_on_wire(Vector2(simWidth/2,simHeight/2),8)
            set_of_balls[i].end_step(dt)

            for j in range(i+1,len(set_of_balls)):
                handleBeadBeadCollision(set_of_balls[i], set_of_balls[j])
            #print(ball.energy)

        pygame.display.flip()
        clock.tick(FPS)
gravity = Vector2(0.0,-10.0)
Run(simWidth,FPS)

pygame.quit()
sys.exit()