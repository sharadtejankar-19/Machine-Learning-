import numpy as np
import matplotlib.pyplot as plt


maze = np.array([
 [0,0,0,1,0],
 [1,1,0,1,0],
 [0,0,0,0,0],
 [0,1,1,1,1],
 [0,0,0,0,0]
])

start, goal = (0,0), (4,4)
actions = [(-1,0),(1,0),(0,-1),(0,1)]
Q = np.zeros((5,5,4))

for _ in range(500):
    s = start
    while s != goal:
        r,c = s
        a = np.random.randint(4) if np.random.rand()<0.2 else np.argmax(Q[r,c])
        dr,dc = actions[a]
        nr,nc = r+dr,c+dc

        if 0<=nr<5 and 0<=nc<5 and maze[nr,nc]==0:
            reward = 100 if (nr,nc)==goal else -1
            Q[r,c,a] += 0.1*(reward + 0.9*np.max(Q[nr,nc])-Q[r,c,a])
            s = (nr,nc)

path = [start]
s = start

while s != goal:
    r,c = s
    a = np.argmax(Q[r,c])
    dr,dc = actions[a]
    s = (r+dr,c+dc)
    path.append(s)

print("Path:",path)

plt.imshow(maze)
plt.plot([p[1] for p in path],[p[0] for p in path])
plt.title("Q-Learning Maze")
plt.show()