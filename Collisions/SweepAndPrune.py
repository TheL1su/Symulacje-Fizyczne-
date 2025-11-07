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


def sweep_and_prune(Spheres):
    intervals = [(sphere, sphere.left(), sphere.right()) for sphere in Spheres]
    intervals.sort(key=lambda x: x[1])
    for i in range(len(Spheres)):
        sp1 = intervals[i][0]
        sp1_x_max = intervals[i][2]
        for j in range(i+1,len(Spheres)):
            sp2 = intervals[j][0]
            sp2_x_min = intervals[j][1]
            if(sp1_x_max <= sp2_x_min):
                break
            if (sp1.bottom() <= sp2.top() and sp2.bottom() <= sp1.top() and
                sp1.front() <= sp2.back() and sp2.front() <= sp1.back()):
                check_collision(sp1,sp2)


Spheres = [SphereObject(i) for i in range(SPHERE_COUNT)]

while True:
    rate(120)
    frame_count += 1

    COLLISION_COUNT = 0
    for sphere in Spheres:
        sphere.update()
    for sphere in Spheres:
        sphere.reset_collision()
    sweep_and_prune(Spheres)

    now = time.time()
    if now - last_time >= 1.0:
        fps = frame_count / (now - last_time)
        frame_count = 0
        last_time = now

    info_label.text = f"FPS: {fps} | Collisions: {COLLISION_COUNT}"

