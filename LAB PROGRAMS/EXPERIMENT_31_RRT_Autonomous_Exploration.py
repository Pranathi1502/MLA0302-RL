import numpy as np
import matplotlib.pyplot as plt

print("==============================================")
print("EXPERIMENT 31 - RRT AUTONOMOUS EXPLORATION")
print("==============================================")

# -------------------------------------------------
# 1. Environment
# -------------------------------------------------

start = np.array([5.0, 5.0])
goal = np.array([95.0, 95.0])

# Obstacles: [x, y, width, height]
obstacles = [
    [20, 15, 20, 25],
    [55, 10, 25, 20],
    [35, 55, 25, 20],
    [70, 60, 15, 25]
]

# -------------------------------------------------
# 2. Collision checking
# -------------------------------------------------

def collision(point):

    for x, y, w, h in obstacles:

        if (
            x <= point[0] <= x + w
            and
            y <= point[1] <= y + h
        ):
            return True

    return False


def line_collision(p1, p2):

    distance = np.linalg.norm(p2 - p1)

    steps = max(int(distance / 1.0), 1)

    for i in range(steps + 1):

        t = i / steps

        point = p1 + t * (p2 - p1)

        if collision(point):
            return True

    return False


# -------------------------------------------------
# 3. RRT parameters
# -------------------------------------------------

nodes = [start]
parents = [-1]

step_size = 3.0
max_iterations = 5000
goal_threshold = 4.0

goal_reached = False


# -------------------------------------------------
# 4. RRT Algorithm
# -------------------------------------------------

for iteration in range(max_iterations):

    # Random point
    random_point = np.random.uniform(
        0, 100, 2
    )

    # Find nearest node
    distances = [
        np.linalg.norm(node - random_point)
        for node in nodes
    ]

    nearest_index = np.argmin(distances)

    nearest_node = nodes[nearest_index]

    # Direction
    direction = random_point - nearest_node

    distance = np.linalg.norm(direction)

    if distance == 0:
        continue

    direction = direction / distance

    # Generate new node
    new_node = (
        nearest_node
        + direction * min(step_size, distance)
    )

    # Keep inside environment
    if not (
        0 <= new_node[0] <= 100
        and
        0 <= new_node[1] <= 100
    ):
        continue

    # Check collision
    if line_collision(nearest_node, new_node):
        continue

    # Add node
    nodes.append(new_node)
    parents.append(nearest_index)

    # Check goal
    if np.linalg.norm(new_node - goal) < goal_threshold:

        if not line_collision(new_node, goal):

            nodes.append(goal)
            parents.append(len(nodes) - 2)

            goal_reached = True

            print("\nGoal reached!")
            print("Iterations:", iteration + 1)

            break


# -------------------------------------------------
# 5. Generate path
# -------------------------------------------------

path = []

if goal_reached:

    current = len(nodes) - 1

    while current != -1:

        path.append(nodes[current])

        current = parents[current]

    path.reverse()

    path = np.array(path)

else:

    print("\nGoal was not reached.")
    path = np.array([start])


# -------------------------------------------------
# 6. Display results
# -------------------------------------------------

print("\nTotal RRT Nodes:", len(nodes))

if goal_reached:

    print("Path Points:", len(path))

    print("\nOptimal Exploration Path:")

    for i, point in enumerate(path):

        print(
            "Point",
            i + 1,
            ":",
            "(",
            round(point[0], 2),
            ",",
            round(point[1], 2),
            ")"
        )

else:

    print("No collision-free path found.")


# -------------------------------------------------
# 7. Visualization
# -------------------------------------------------

plt.figure(figsize=(8, 8))

# Draw obstacles
for x, y, w, h in obstacles:

    rectangle = plt.Rectangle(
        (x, y),
        w,
        h
    )

    plt.gca().add_patch(rectangle)


# Draw RRT tree
for i in range(1, len(nodes)):

    parent = parents[i]

    plt.plot(
        [nodes[i][0], nodes[parent][0]],
        [nodes[i][1], nodes[parent][1]],
        "gray",
        linewidth=0.5
    )


# Draw final path
if goal_reached:

    plt.plot(
        path[:, 0],
        path[:, 1],
        "b-",
        linewidth=3,
        label="RRT Path"
    )


# Start
plt.plot(
    start[0],
    start[1],
    "go",
    markersize=10,
    label="Start"
)


# Goal
plt.plot(
    goal[0],
    goal[1],
    "ro",
    markersize=10,
    label="Goal"
)


plt.xlim(0, 100)
plt.ylim(0, 100)

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.title(
    "RRT Autonomous Exploration Robot"
)

plt.legend()
plt.grid()

plt.show()

print("\nExperiment 31 Completed Successfully!")