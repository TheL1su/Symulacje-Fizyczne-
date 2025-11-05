from vpython import box, vector, color, rate, scene, label
import random
import math
import time

WORLD_SIZE = 20
BOX_COUNT = 500
MIN_BOX_SIZE = 0.5
MAX_BOX_SIZE = 1.5
MAX_SPEED = 0.05

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

class BoxObject:
    def __init__(self, idx):
        self.id = idx

        # randomowe pozycje boxow
        self.size = vector(
            random.uniform(MIN_BOX_SIZE, MAX_BOX_SIZE),
            random.uniform(MIN_BOX_SIZE, MAX_BOX_SIZE),
            random.uniform(MIN_BOX_SIZE, MAX_BOX_SIZE),
        )
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

        self.visual = box(pos=self.pos, size=self.size, color=color.white)

    #funkcja do poruszania sie
    def update(self):
        self.pos += self.vel
        self.visual.pos = self.pos

        #dla kazdej wspolrzednej sprawdzamy czy nie wyszlismy poza obszar
        for cord in ["x", "y", "z"]:
            val = getattr(self.pos, cord)
            half = getattr(self.size, cord) / 2
            if abs(val) + half > WORLD_SIZE / 2:
                setattr(self.vel, cord, -getattr(self.vel, cord))
                setattr(self.pos, cord, math.copysign(WORLD_SIZE / 2 - half, val))
                self.visual.pos = self.pos

    def AABB(self):
        return {
            "min": (self.pos.x - self.size.x / 2,
                    self.pos.y - self.size.y / 2,
                    self.pos.z - self.size.z / 2),
            "max": (self.pos.x + self.size.x / 2,
                    self.pos.y + self.size.y / 2,
                    self.pos.z + self.size.z / 2)
        }

    def check_collision(self, other):
        first = self.AABB()
        second = other.AABB()
        return (first["min"][0] <= second["max"][0] and first["max"][0] >= second["min"][0] and
                first["min"][1] <= second["max"][1] and first["max"][1] >= second["min"][1] and
                first["min"][2] <= second["max"][2] and first["max"][2] >= second["min"][2])

class BVHNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.box_id = -1
        self.aabb = {"min": [float("inf")] * 3, "max": [float("-inf")] * 3}

    def is_leaf(self):
        return self.box_id != -1

def aabb_intersect(a, b):
    return all(a["min"][i] <= b["max"][i] and a["max"][i] >= b["min"][i] for i in range(3))

def create_leaf(box_id, box):
    node = BVHNode()
    node.box_id = box_id
    aabb = box.AABB()
    node.aabb["min"] = list(aabb["min"])
    node.aabb["max"] = list(aabb["max"])
    return node

def create_subtree(boxes_list, begin, end, boxes):
    if begin == end:
        return create_leaf(boxes_list[begin]["id"], boxes[boxes_list[begin]["id"]])
    mid = (begin + end) // 2
    node = BVHNode()
    node.left = create_subtree(boxes_list, begin, mid, boxes)
    node.right = create_subtree(boxes_list, mid + 1, end, boxes)
    for i in range(3):
        node.aabb["min"][i] = min(node.left.aabb["min"][i], node.right.aabb["min"][i])
        node.aabb["max"][i] = max(node.left.aabb["max"][i], node.right.aabb["max"][i])
    return node

def create_tree(boxes):
    box_list = [{"id": i, "key": i} for i in range(len(boxes))]
    return create_subtree(box_list, 0, len(box_list) - 1, boxes)

def find_collisions(box_id, box, node, boxes, collisions):
    if not aabb_intersect(box.AABB(), node.aabb):
        return
    if node.is_leaf():
        if node.box_id != box_id and box.check_collision(boxes[node.box_id]):
            collisions.append((box_id, node.box_id))
    else:
        find_collisions(box_id, box, node.left, boxes, collisions)
        find_collisions(box_id, box, node.right, boxes, collisions)

def check_collisions_bvh(boxes, root):
    collisions = []
    for i, box in enumerate(boxes):
        find_collisions(i, box, root, boxes, collisions)
    return collisions

boxes = [BoxObject(i) for i in range(BOX_COUNT)]

while True:
    rate(200)
    frame_count += 1
    for box in boxes:
        box.update()

    root = create_tree(boxes)
    collisions = check_collisions_bvh(boxes, root)


    for element in boxes:
        element.visual.color = color.white

    for a, b in collisions:
        boxes[a].visual.color = color.red
        boxes[b].visual.color = color.red

    now = time.time()
    if now - last_time >= 1.0:
        fps = frame_count / (now - last_time)
        frame_count = 0
        last_time = now

    info_label.text = f"FPS: {fps} | Collisions: {len(collisions)}"
