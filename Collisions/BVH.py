from vpython import *
import math
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

def normalize_coord(val):
    return (val + WORLD_SIZE / 2) / WORLD_SIZE

def expand_bits(v):
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v

def calculate_morton_code(position):
    
    x = normalize_coord(position.x)
    y = normalize_coord(position.y)
    z = normalize_coord(position.z)
    
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    z = min(max(z, 0.0), 1.0)
    
    x = min(math.floor(x * 1023), 1023)
    y = min(math.floor(y * 1023), 1023)
    z = min(math.floor(z * 1023), 1023)
    
    xx = expand_bits(x)
    yy = expand_bits(y)
    zz = expand_bits(z)
    
    return xx | (yy << 1) | (zz << 2)

class BVHNode:
    def __init__(self):
        self.sphere = None
        self.minimum = vector(math.inf, math.inf, math.inf)
        self.maximum = vector(-math.inf, -math.inf, -math.inf)

    def is_leaf(self):
        return self.sphere is not None

def create_tree(spheres):
    list_of_spheres = [ {"sphere" : sphere, "morton_code" : calculate_morton_code(sphere.pos) } for sphere in spheres ]
    list_of_spheres.sort(key = lambda x: x["morton_code"])

    return create_subtree(list_of_spheres, 0, len(list_of_spheres)-1)




def create_subtree(spheres, begin, end):
    if begin == end:
        return create_leaf(spheres[begin])

    else:
        middle = (begin+end)//2
        node = BVHNode()

        node.left = create_subtree(spheres, begin, middle)
        node.right = create_subtree(spheres, middle+1, end)

        node.minimum.x = min(node.left.minimum.x, node.right.minimum.x)
        node.minimum.y = min(node.left.minimum.y, node.right.minimum.y)
        node.minimum.z = min(node.left.minimum.z, node.right.minimum.z)

        node.maximum.x = max(node.left.maximum.x, node.right.maximum.x)
        node.maximum.y = max(node.left.maximum.y, node.right.maximum.y)
        node.maximum.z = max(node.left.maximum.z, node.right.maximum.z)

        return node

def create_leaf(dict):
    node = BVHNode()
    node.sphere = dict["sphere"]
    node.minimum = node.sphere.minimum()
    node.maximum = node.sphere.maximum()
    return node

def check_boxes_intersect(box1_min, box1_max, box2_min, box2_max):
    return (box1_min.x <=box2_max.x and box2_min.x <= box1_max.x
            and box1_min.y <=box2_max.y and box2_min.y <= box1_max.y
            and box1_min.z <=box2_max.z and box2_min.z <= box1_max.z)


def find_collisions(sphere, minimum, maximum, node):
    if check_boxes_intersect(minimum, maximum, node.minimum, node.maximum):
        if node.is_leaf():
            if node.sphere != sphere:
                check_collision(sphere, node.sphere)

        else:
            find_collisions(sphere, minimum, maximum, node.left)
            find_collisions(sphere, minimum, maximum, node.right)

def check_collision(sphere1,sphere2):
    radius =sphere1.getradius()
    radius2 =sphere2.getradius()
    if mag(sphere1.pos - sphere2.pos) < radius + radius2:
        global COLLISION_COUNT
        COLLISION_COUNT += 1
        sphere1.collision()
        sphere2.collision()

def bvh_collisions(spheres):
    tree = create_tree(spheres)
    for sphere in spheres:
        if not sphere.collided():
            find_collisions(sphere, sphere.minimum(), sphere.maximum(), tree)

Spheres = [SphereObject(i) for i in range(SPHERE_COUNT)]

while True:
    rate(120)
    frame_count += 1

    COLLISION_COUNT = 0
    for sphere in Spheres:
        sphere.update()
    for sphere in Spheres:
        sphere.reset_collision()
    bvh_collisions(Spheres)

    now = time.time()
    if now - last_time >= 1.0:
        fps = frame_count / (now - last_time)
        frame_count = 0
        last_time = now

    info_label.text = f"FPS: {fps} | Collisions: {COLLISION_COUNT}"
