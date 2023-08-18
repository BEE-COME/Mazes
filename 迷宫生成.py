
from PIL import Image, ImageDraw
import random
from collections import deque
from queue import Queue
import time

import matplotlib.pyplot as plt

import threading

import heapq
#地图长宽
row=1000
col=1000

def heuristic(node, goal):
    # 使用曼哈顿距离作为启发函数
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def find_shortest_path2(maze):
    start = (0, 0)  # 入口坐标
    end = (len(maze)-1, len(maze[0])-1)  # 出口坐标

    # 创建一个优先队列来存储待探索的节点
    queue = [(0, start)]
    heapq.heapify(queue)

    # 创建一个字典来存储每个节点的最短路径距离
    distance = {start: 0}

    # 创建一个字典来存储每个节点的前一个节点，用于最后重构路径
    previous = {start: None}

    while queue:
        # 从队列中取出距离最小的节点
        current_distance, current = heapq.heappop(queue)

        # 到达目标位置，结束搜索
        if current == end:
            break

        # 检查上下左右四个方向的邻居
        neighbors = [(current[0]-1, current[1]), (current[0]+1, current[1]), (current[0], current[1]-1), (current[0], current[1]+1)]
        for neighbor in neighbors:
            n_row, n_col = neighbor

            # 检查邻居是否在迷宫范围内且不是墙壁
            if 0 <= n_row < len(maze) and 0 <= n_col < len(maze[0]) and maze[n_row][n_col] == 0:
                # 计算邻居节点的新距离
                new_distance = distance[current] + 1

                # 如果新距离比之前的距离更小，更新距离和前一个节点
                if neighbor not in distance or new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    previous[neighbor] = current

                    # 计算启发函数的值
                    priority = new_distance + heuristic(neighbor, end)

                    # 将邻居节点加入优先队列
                    heapq.heappush(queue, (priority, neighbor))

    # 重构路径
    path = []
    if end in previous:
        current = end
        while current:
            path.append(current)
            current = previous[current]
        path.reverse()

    return path

# 生成迷宫
def generate_maze(rows, cols, num_paths):
        
    # 创建一个二维列表来表示迷宫，初始值都为1，表示墙壁
    maze = [[1] * cols for _ in range(rows)]
    
    # 创建一个栈来保存已经访问过的节点
    stack = []
    
    # 随机选择起点
    start_row, start_col = 0, 0
    
    # 将起点设为通路（值为0）
    maze[start_row][start_col] = 0
    
    # 将起点加入栈中
    stack.append((start_row, start_col))

    # 当栈不为空时，继续探索迷宫
    while stack:
        current_row, current_col = stack[-1]
        
        # 存储未访问的邻居
        neighbors = []

        # 寻找未访问的邻居
        if current_row > 1 and maze[current_row-2][current_col] == 1:
            neighbors.append((current_row-2, current_col))
        if current_row < rows-2 and maze[current_row+2][current_col] == 1:
            neighbors.append((current_row+2, current_col))
        if current_col > 1 and maze[current_row][current_col-2] == 1:
            neighbors.append((current_row, current_col-2))
        if current_col < cols-2 and maze[current_row][current_col+2] == 1:
            neighbors.append((current_row, current_col+2))

        if neighbors:
            # 随机选择一个邻居并打通墙壁
            next_row, next_col = random.choice(neighbors)
            maze[next_row][next_col] = 0
            maze[(current_row+next_row)//2][(current_col+next_col)//2] = 0
            stack.append((next_row, next_col))
        else:
            # 没有未访问的邻居，回溯
            stack.pop()  

    # for x in range(rows):
    #     for y in range(cols):
    #         if(maze[x][y]>0):
    #             if(random.randrange(0,100)>60):
    #                 maze[x][y]-=1

    # 将起点和终点设为通路
    maze[0][0] = 0
    maze[rows-1][cols-1] = 0
    maze[rows-2][cols-1] = 0
    maze[rows-1][cols-2] = 0
    
    # 返回生成的迷宫
    return maze

def search(maze, start, end, path, solutions):
    row, col = start

    # 到达出口，将路径添加到解决方案列表
    if start == end:
        solutions.append(path + [end])
        return

    # 标记当前坐标为已访问
    maze[row][col] = 1

    # 检查上下左右四个方向的邻居
    neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
    for neighbor in neighbors:
        n_row, n_col = neighbor

        # 检查邻居是否在迷宫范围内且不是墙壁，并且没有被访问过
        if 0 <= n_row < len(maze) and 0 <= n_col < len(maze[0]) and maze[n_row][n_col] == 0:
            # 创建新线程进行搜索
            thread = threading.Thread(target=search, args=(maze.copy(), neighbor, end, path + [start], solutions))
            thread.start()

def find_shortest_path1(maze):
    start = (0, 0)  # 入口坐标
    end = (len(maze)-1, len(maze[0])-1)  # 出口坐标

    solutions = []  # 存储解决方案的列表

    # 开始搜索
    search(maze, start, end, [], solutions)

    # 等待所有线程结束
    for thread in threading.enumerate():
        if thread != threading.main_thread():
            thread.join()

    if solutions:
        # 选择最短路径
        shortest_path = min(solutions, key=len)
        return shortest_path
    else:
        return None  # 如果找不到最佳路径，返回None
    
def find_shortest_path(maze):

    start = (0, 0)  # 入口坐标
    end = (len(maze)-1, len(maze[0])-1)  # 出口坐标

    visited = set()  # 记录已经访问过的坐标
    queue = Queue()  # 广度优先搜索的队列
    queue.put((start, []))  # 将起始坐标和路径加入队列

    while not queue.empty():
        current, path = queue.get()
        row, col = current

        # 到达出口，返回路径
        if current == end:
            return path + [current]

        # 标记当前坐标为已访问
        visited.add(current)

        # 检查上下左右四个方向的邻居
        neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        for neighbor in neighbors:
            n_row, n_col = neighbor

            # 检查邻居是否在迷宫范围内且不是墙壁，并且没有被访问过
            if 0 <= n_row < len(maze) and 0 <= n_col < len(maze[0]) and maze[n_row][n_col] == 0 and neighbor not in visited:
                queue.put((neighbor, path + [current]))  # 将邻居加入队列，并将路径更新为当前路径加上当前坐标
                visited.add(neighbor)  # 标记邻居为已访问

    return None  # 如果找不到最佳路径，返回None

start_time = time.time()  # 记录开始时间
# 生成迷宫
maze = generate_maze(row, col,5)#迷宫大小
# for row in maze:
#     print(' '.join(str(cell) for cell in row))

# 定义单元格的大小和墙壁的宽度
cell_size = 1
wall_width = 1


# for row in maze:
#     print(' '.join(map(str, row)))
# print(maze)

# 计算图像的大小
maze_width = len(maze[0])
maze_height = len(maze)
image_width = maze_width * cell_size
image_height = maze_height * cell_size

# 创建空白图像
image = Image.new("RGB", (image_width, image_height), "white")
draw = ImageDraw.Draw(image)

# 绘制迷宫的墙壁和路径
for row in range(maze_height):
    for col in range(maze_width):
        x1 = col * cell_size
        y1 = row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size

        if maze[row][col] == 1:
            draw.rectangle((x1, y1, x2, y2), fill="black")
        else:
            draw.rectangle((x1, y1, x2, y2), fill="white")

# 标记入口和出口
draw.rectangle((0, 0, cell_size, cell_size), fill="green")
draw.rectangle((image_width - cell_size, image_height - cell_size, image_width, image_height), fill="red")
end_time = time.time()  # 记录结束时间
print("地图生成时间：", end_time - start_time, "秒")
image.save("1.png")
# image.save("1.png", quality=1)
# # 找到最佳路径
# start_time = time.time()  # 记录开始时间

# path = find_shortest_path(maze)  # 找到最佳路径

# end_time = time.time()  # 记录结束时间

# # 将最佳路径以绿线标记在图像上
# try:
#     for i in range(len(path) - 1):
#         x1 = path[i][1] * cell_size + cell_size // 2
#         y1 = path[i][0] * cell_size + cell_size // 2
#         x2 = path[i+1][1] * cell_size + cell_size // 2
#         y2 = path[i+1][0] * cell_size + cell_size // 2

#         draw.line((x1, y1, x2, y2), fill="green", width=20)
# except :
#     print("error")

# finally:
# # 保存图像
#     image.save("1.png")
# print("运行时间：", end_time - start_time, "秒")



# # 找到最佳路径
# start_time = time.time()  # 记录开始时间

# path = find_shortest_path2(maze)  # 找到最佳路径

# end_time = time.time()  # 记录结束时间

# # 将最佳路径以绿线标记在图像上
# try:
#     for i in range(len(path) - 1):
#         x1 = path[i][1] * cell_size + cell_size // 2
#         y1 = path[i][0] * cell_size + cell_size // 2
#         x2 = path[i+1][1] * cell_size + cell_size // 2
#         y2 = path[i+1][0] * cell_size + cell_size // 2

#         draw.line((x1, y1, x2, y2), fill="red", width=20)
# except :
#     print("error")

# finally:
# # 保存图像
#     image.save("1.png")
# print("运行时间：", end_time - start_time, "秒")
