from vpython import *
import time
from Sphere import SphereObject

WORLD_SIZE = 20
SPHERE_COUNT = 500
COLLISION_COUNT = 0
scene.width = 900
scene.height = 700
scene.title = "BVH - 3D Box Collision Demo"
scene.center = vector(0, 0, 0)
scene.background = color.gray(0.1)

info_label = label(
    pos=vector(0, WORLD_SIZE/2 + 2, 0),
    text="Initializing...",
    height=20,
    box=False,
    color=color.white,
    align="center",
)
frame_count = 0
last_time = time.time()
fps = 0


def check_collision(sphere1,sphere2):
    radius =sphere1.getradius()
    radius2=sphere2.getradius()
    if mag(sphere1.pos - sphere2.pos) < radius + radius2:
        global COLLISION_COUNT
        COLLISION_COUNT += 1
        sphere1.collision()
        sphere2.collision()


def find_collisions(Spheres):
    for i in range(0,len(Spheres)):
        for j in range(i+1,len(Spheres)):
            check_collision(Spheres[i],Spheres[j])


Spheres = [SphereObject(i) for i in range(SPHERE_COUNT)]

while True:
    rate(120)
    frame_count += 1

    COLLISION_COUNT = 0
    for sphere in Spheres:
        sphere.update()
    for sphere in Spheres:
        sphere.reset_collision()
    find_collisions(Spheres)

    now = time.time()
    if now - last_time >= 1.0:
        fps = frame_count / (now - last_time)
        frame_count = 0
        last_time = now

    info_label.text = f"FPS: {fps} | Collisions: {COLLISION_COUNT}"



