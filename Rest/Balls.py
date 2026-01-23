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
timeStep = 1/60.0
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Ball simulation")
clock = pygame.time.Clock()
gravity = [0.0,-10.0]
drag = 0.995

def cX(x):
    return int(x * cScale)

def cY(y):
    return int(WINDOW_HEIGHT - y * cScale)

def Run(timeStep,gravity,simWidth,FPS):
    rectangle = Rectangle([simWidth / 2 - 2, simHeight / 2], 4, 3)

    set_of_balls = []
    for _ in range(0,15):
        radius = random.uniform(0.1, 1)
        pos = [random.uniform(1,simWidth),random.uniform(1,simHeight-2)]
        vel = [random.uniform(-15,15),random.uniform(-15,15)]
        colour = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
        set_of_balls.append(Ball(radius,pos,vel,colour))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    for ball in set_of_balls:
                        ball.boost_up()
                elif event.key == pygame.K_LEFT:
                    for ball in set_of_balls:
                        ball.boost_left()
                elif event.key == pygame.K_RIGHT:
                    for ball in set_of_balls:
                        ball.boost_right()
                elif event.key == pygame.K_DOWN:
                    for ball in set_of_balls:
                        ball.boost_down()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for ball in set_of_balls:
                    ball.boost_up()

        screen.fill((255, 255, 255))
        rectangle.draw()
        for ball in set_of_balls:
            ball.update(timeStep,gravity)
            ball.reflect(simWidth)
            ball.collide_with_rectangle(rectangle)
            ball.draw_circle()
            #print(ball.energy)

        pygame.display.flip()
        clock.tick(FPS)



class Ball:

    def __init__(self,radius,pos,vel,colour):
        self.radius = radius
        self.pos = pos
        self.vel = vel
        self.col = colour
        self.energy = (1/2)*np.linalg.norm(pos)**2 - gravity[1]*pos[1]

    def update(self,timeStep,gravity):
        vel_old_x = self.vel[0]
        vel_old_y = self.vel[1]
        vel_x = (self.vel[0] + gravity[0] * timeStep)*drag
        vel_y = (self.vel[1] + gravity[1] * timeStep)*drag
        self.vel = [vel_x,vel_y]

        pos_x = self.pos[0] + ((vel_x + vel_old_x)/2)*timeStep
        pos_y = self.pos[1] + ((vel_y + vel_old_y)/2)*timeStep
        self.pos = [pos_x,pos_y]

        self.energy = (1/2)*np.linalg.norm(self.vel)**2 - gravity[1]*self.pos[1]

    def reflect(self,simWidth):
        #x
        if self.pos[0] <= self.radius:
            self.pos[0] = -self.pos[0] + 2*self.radius
            self.vel[0] = -self.vel[0]*drag

        if self.pos[0] >= simWidth-self.radius:
            self.pos[0] = simWidth - (self.pos[0] - simWidth) - 2*self.radius
            self.vel[0] = -self.vel[0]*drag
        #y
        if self.pos[1] <= self.radius:
            self.pos[1] = self.radius
            self.vel[1] = np.sqrt((2*self.energy + 2*self.pos[1]*gravity[1]) - self.vel[0]**2)*drag
    
    def collide_with_rectangle(self, rectangle):
        left = rectangle.pos[0]
        right = rectangle.pos[0] + rectangle.width
        bottom = rectangle.pos[1]
        top = rectangle.pos[1] + rectangle.height


        x, y = self.pos
        r = self.radius

        nearest_x = max(left, min(x, right))
        nearest_y = max(bottom, min(y, top))

        nearest_vec_x = x - nearest_x
        nearest_vec_y = y - nearest_y
        dist = np.linalg.norm([nearest_vec_x,nearest_vec_y])

        if dist < r:
            norm_x = nearest_vec_x / dist
            norm_y = nearest_vec_y / dist

            overlap = r - dist
            self.pos[0] += norm_x * overlap
            self.pos[1] += norm_y * overlap

            v_dot_n = self.vel[0] * norm_x + self.vel[1] * norm_y
            self.vel[0] -= 2 * v_dot_n * norm_x * drag
            self.vel[1] -= 2 * v_dot_n * norm_y * drag

    def draw_circle(self):
        pygame.draw.circle(screen,self.col,
        (cX(self.pos[0]), cY(self.pos[1])),
        int(cScale * self.radius))


    def boost_up(self):
        self.vel[1] += 5
    def boost_down(self):
        self.vel[1] -= 5
    def boost_right(self):
        self.vel[0] += 5
    def boost_left(self):
        self.vel[0] -= 5

class Rectangle:

    def __init__(self, pos, width, height, color=(0, 0, 0)):
        self.pos = pos
        self.width = width
        self.height = height
        self.color = color

    def draw(self):
        pygame.draw.rect(screen, self.color, 
        (cX(self.pos[0]), cY(self.pos[1] + self.height), 
        cScale * self.width, cScale * self.height), width=2)



Run(timeStep,gravity,simWidth,FPS)

pygame.quit()
sys.exit()

 