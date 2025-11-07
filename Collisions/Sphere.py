from vpython import sphere,vector,color,mag
import random
import math
import time
TIME_STEP = 1.0 / 60.0
WORLD_SIZE = 20
MINradius = 0.5
MAXradius = 1.5
MAX_SPEED = 0.05

class SphereObject:
    def __init__(self, idx):
        self.id = idx

        # randomowe pozycje boxow
        self.radius = random.uniform(MINradius, MAXradius)
        self.pos = vector(
            random.uniform(-WORLD_SIZE / 2, WORLD_SIZE / 2),
            random.uniform(-WORLD_SIZE / 2, WORLD_SIZE / 2),
            random.uniform(-WORLD_SIZE / 2, WORLD_SIZE / 2),
        )
        self.vel = vector(
            random.uniform(-MAX_SPEED, MAX_SPEED),
            random.uniform(-MAX_SPEED, MAX_SPEED),
            random.uniform(-MAX_SPEED, MAX_SPEED),
        )
        self.visual = sphere(pos=self.pos, radius=self.radius, color=color.white)

    def left(self):
        return self.pos.x - self.radius
    
    def right(self):
        return self.pos.x + self.radius
    
    def top(self):
        return self.pos.y + self.radius
    
    def bottom(self):
        return self.pos.y - self.radius
    
    def front(self):
        return self.pos.z - self.radius
    
    def back(self):
        return self.pos.z + self.radius
    
    def minimum(self):
        return vector(self.left(), self.bottom(), self.front())
    
    def maximum(self):
        return vector(self.right(), self.top(), self.back())
    
    def getradius(self):
        return self.radius
    
    def collision(self):
        self.visual.color = color.red

    def reset_collision(self):
        self.visual.color = color.white
    
    def update(self):
        self.move()
        self.bounce()
    def collided(self):
        return self.visual.color == color.red

    def move(self):
        self.old_pos = self.pos
        self.pos += self.vel
        self.visual.pos = self.pos

    def bounce(self):
        boundary = WORLD_SIZE/2

        #Bottom
        if self.pos.y <= -boundary + self.radius:
            self.pos.y = -boundary + self.radius
            self.vel.y = -self.vel.y

        #Top
        elif self.pos.y >= boundary - self.radius:
            self.pos.y = boundary - self.radius
            self.vel.y = -self.vel.y

        #Left
        if self.pos.x <= -boundary + self.radius:
            self.pos.x = -self.pos.x + 2*(self.radius -boundary)
            self.vel.x = -self.vel.x

        #Right
        elif self.pos.x >= boundary - self.radius:
            self.pos.x = -self.pos.x + 2*(boundary - self.radius)
            self.vel.x = -self.vel.x

        # Front
        if self.pos.z <= -boundary + self.radius:
            self.pos.z = -boundary + self.radius
            self.vel.z = -self.vel.z

        # Back
        elif self.pos.z >= boundary - self.radius:
            self.pos.z = boundary - self.radius
            self.vel.z = -self.vel.z
