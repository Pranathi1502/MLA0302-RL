import numpy as np
import matplotlib.pyplot as plt

start = np.array([10, 10])
goal = np.array([90, 90])

obstacles = [
    [25, 20, 20, 30],
    [55, 15, 20, 25],
    [35, 60, 25, 20],
    [70, 55, 15, 25]
]

nodes = [start]
parents = [-1]

def collision(p):
    for x, y, w, h in obstacles:
        if x <= p[0] <= x + w and y <= p[1] <= y + h:
            return True
    return False

for _ in range(3000):

    point = np.random.randint(0, 101, 2)

    distances = [np.linalg.norm(n - point) for n in nodes]
    nearest_index = np.argmin(distances)
    nearest = nodes[nearest_index]

    direction = point - nearest
    distance = np.linalg.norm(direction)

    if distance == 0:
        continue

    new = nearest + direction / distance * min(3, distance)

    if not collision(new):
        nodes.append(new)
        parents.append(nearest_index)

        if np.linalg.norm(new - goal) < 3:
            nodes.append(goal)
            parents.append(len(nodes) - 2)
            break

path = []
i = len(nodes) - 1

while i != -1:
    path.append(nodes[i])
    i = parents[i]

path.reverse()
path = np.array(path)

plt.figure(figsize=(7, 7))

for x, y, w, h in obstacles:
    plt.gca().add_patch(
        plt.Rectangle((x, y), w, h)
    )

plt.plot(path[:, 0], path[:, 1], "b-", linewidth=2)
plt.plot(start[0], start[1], "go", markersize=8)
plt.plot(goal[0], goal[1], "ro", markersize=8)

plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("UAV RRT Path Planning")
plt.grid()
plt.show()

print("RRT Path Planning Completed")
print("Path Points :", len(path))