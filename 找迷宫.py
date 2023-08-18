# -*- coding: utf-8 -*-

import numpy as np
import cv2
from queue import Queue
import time
import heapq
weith=1
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
        neighbors = [(row-weith, col), (row+weith, col), (row, col-weith), (row, col+weith)]
        for neighbor in neighbors:
            n_row, n_col = neighbor

            # 检查邻居是否在迷宫范围内且不是墙壁，并且没有被访问过
            if 0 <= n_row < len(maze) and 0 <= n_col < len(maze[0]) and maze[n_row][n_col] == 0 and neighbor not in visited:
                queue.put((neighbor, path + [current]))  # 将邻居加入队列，并将路径更新为当前路径加上当前坐标
                visited.add(neighbor)  # 标记邻居为已访问

    return None  # 如果找不到最佳路径，返回None

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

# im = cv2.imread('test.jpg')#
im = cv2.imread('1.png')#
# im = cv2.imread('../data/black-white-rect.png')#contour.jpg #
# im = cv2.imread('../data/chessboard.jpeg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# cv2.imshow("imgray", imgray)

#需要注意的是cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）
# 所以读取的图像要先转成灰度的，再转成二值图
# ret, thresh = cv2.threshold(imgray, 0, 25, 0)
ret, thresh = cv2.threshold(imgray, 0, 100, 0)
# ret, thresh = cv2.threshold(src=imgray, thresh=150, maxval=255, type=cv2.THRESH_BINARY)#src, thresh, maxval, type

# cv2.imshow("thresh", thresh)
#轮廓提取模式 Contour_Retrieval_Mode

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("contours size: ", len(contours))
maze_array = np.zeros_like(im)
img = cv2.drawContours(maze_array, contours, -1, (255,255,255), thickness=cv2.FILLED, hierarchy=hierarchy)#获取轮廓和数组
# img = cv2.drawContours(im, contours, 3, (255, 0, 0), 3)

# maze_array = maze_array[:,:,0] // 255
# for row in maze_array:
#     print(' '.join(map(str, row)))


# def print_maze_to_txt(maze_array, file_path):
#     with open(file_path, 'w') as file:
#         for layer in maze_array:
#             for row in layer:
#                 formatted_row = list(map(lambda num: str(num).zfill(3), row))
#                 file.write(' '.join(formatted_row)+"    " )
#             file.write('\n')


# print(maze_array.ndim)
# print(maze_array.shape)
# print(maze_array[0][0][0])
def convert_to_2d_array(maze_array):
    # 获取迷宫数组的维度
    rows, cols,_ = maze_array.shape
    

    # 创建一个新的二维数组
    maze_2d = np.empty((rows, cols), dtype=int)
    # print(maze_2d.shape)

    for i in range(maze_array.shape[0]):
        for j in range(maze_array.shape[1]):
            if maze_array[i][j][0]>0:
                maze_2d[i][j]=0
            else:
                maze_2d[i][j]=1
        # 将数组中的元素根据条件转换为0或1
        # maze_2d = np.where((maze_array == [0, 0, 0]).all(axis=1), 0, 1)

    return maze_2d


maze_2d=convert_to_2d_array(maze_array)
np.savetxt('1.txt', maze_2d, fmt='%d')

# print(maze_2d[0][0])
# for row in maze_2d:
#     print(' '.join(map(str, row)))
# print(maze)



start_time = time.time()  # 记录开始时间


# 找出口
path=find_shortest_path2(maze_2d)
end_time = time.time()  # 记录结束时间
print("运行时间：", end_time - start_time, "秒")
# 将最佳路径以绿线标记在图像上
cell_size=1
try:
    for i in range(len(path) - 1):
        x1 = path[i][1] * cell_size + cell_size // 2
        y1 = path[i][0] * cell_size + cell_size // 2
        x2 = path[i+1][1] * cell_size + cell_size // 2
        y2 = path[i+1][0] * cell_size + cell_size // 2

        # cv2.line(img,(x1, y1, x2, y2), fill="red", width=5)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
except Exception as e:
    print("Error:", e)


# 保存图像

cv2.namedWindow("contour.jpg", 0)
cv2.imshow("contour.jpg", img)
cv2.imwrite("2.png", img)
cv2.waitKey(0)

