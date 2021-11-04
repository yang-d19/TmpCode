from typing import Sequence
import numpy as np
from copy import deepcopy

from numpy.random import shuffle
from utils import *
import time
import math
import random


class Node(object):  # Represents a node in a search tree
    def __init__(self, state, parent=None, action=None, g_cost=0, h_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.path_cost = g_cost + h_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def child_node_with_heuristic(self, problem, action):
        next_state = problem.move(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.g(self.g_cost, self.state, action, next_state),
                         problem.h(self.state)
                         )
        return next_node

    
    def child_node_no_heuristic(self, problem, action):
        next_state = problem.move(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.g(self.g_cost, self.state, action, next_state),
                         0
                         )
        return next_node

    def path(self):
        """
        Returns list of nodes from this node to the root node
        """
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __repr__(self):
        return "<Node {}(g={})>".format(self.state, self.path_cost)

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


class Problem(object):
    def __init__(self, init_state=None, goal_state=None):
        self.init_node = Node(init_state)
        self.goal_node = Node(goal_state)

    def actions(self, state):
        """
        Given the current state, return valid actions.
        :param state:
        :return: valid actions
        """
        pass

    def move(self, state, action):
        pass

    def is_goal(self, state):
        pass

    def g(self, cost, from_state, action, to_state):
        pass
    
    def h(self, this_state):
        pass

    def solution(self, goal):
        """
        Returns actions from this node to the root node
        """
        if goal.state is None:
            return None
        return [node.action for node in goal.path()[1:]]

    def expand_with_heuristic(self, node):  # Returns a list of child nodes
        return [node.child_node_with_heuristic(self, action) for action in self.actions(node.state)]

    def expand_no_heuristic(self, node):  # Returns a list of child nodes
        return [node.child_node_no_heuristic(self, action) for action in self.actions(node.state)]
    

class LinkGame(Problem):

    def __init__(self, height, width, kinds):
        #init_state = self.generate_borad(height, width, kinds)
        #goal_state = self.generate_borad(empty=True)
        self.height = height
        self.width = width
        self.kinds = kinds
        self.poslist = list()

        #super().__init__(init_state, goal_state)


    def generateBorad(self, empty=True):
        # 游戏棋盘
        board = np.zeros((self.height, self.width), dtype=int)
        # 在核心棋盘的四周添加了空白的边，方便搜索路径
        full_board = np.zeros((self.height + 2, self.width + 2), dtype=int)
        # 生成空棋盘
        if empty == True:
            return full_board

        # print(board.shape[0], board.shape[1])

        seq = list()
        # 棋盘总块数
        block_sum = self.height * self.width

        lower = 1
        upper = math.floor(math.floor(block_sum / self.kinds) / 2)
        cnt = 0

        for idx in range(1, self.kinds + 1):
            # 第 idx 种类型的方块的数量
            num = 2 * random.randint(lower, upper)
            # 当前已经生成的块数
            cnt += num
            for i in range(num):
                seq.append(idx)

        # 空的位置的数量
        vacantNum = block_sum - cnt
        for i in range(vacantNum):
            seq.append(0)

        # 随机打乱
        np.random.shuffle(seq)

        for row in range(self.height):
            for col in range(self.width):
                index = row * self.width + col;
                board[row, col] = seq[index]
        
        
        full_board[1: self.height + 1, 1: self.width + 1] = board

        return full_board
        # return full_board.tolist()
    

    def sliceBoard(self, state):
        # 0号即空位置
        # 寻找第 idx 号图案的位置
        for idx in range(0, self.kinds + 1):
            curr_list = []
            corres_rows = np.where(np.array(state) == idx)[0]
            corres_cols = np.where(np.array(state) == idx)[1]
            assert len(corres_rows) == len(corres_cols), "对应图案的行数和列数不等"

            for i in range(len(corres_rows)):
                row = corres_rows[i]
                col = corres_cols[i]
                curr_list.append((row, col))
            # 当前图案的所有位置
            self.poslist.append(curr_list)
    

    # 该位置是否在棋盘内部（不包括周围一圈）
    def isInsideBoard(self, pos):
        (x, y) = pos
        h = self.height + 2
        w = self.width + 2
        return (x > 0) and (x < h - 1) and (y > 0) and (y < w - 1)


    # 找当前给定的块到相同图案的其它块的折数
    # 对于普通规则，折数最大为2
    # 对于扩展规则，折数无最大限制
    def findPath(self, index, pos, state):
        # 扩展过的棋盘大小
        h = state.shape[0]
        w = state.shape[1]
        turn_cnt_map = np.zeros((h, w))
        # 由于我的设置是 -1 表示 EMPTY
        # 所有 map 所有元素减去 1
        turn_cnt_map -= 1

        # const variable
        # 特殊块都是小于 0 的
        EMPTY = -1
        OTHER = -2
        SAME = -3
        # 大于等于 0 的块显示的是折数

        for idx in range(1, self.kinds + 1):
            # 是当前要找的同图案的块
            if idx == index:
                for (x, y) in self.poslist[idx]:
                    # 在矩阵中用 -3 表示
                    turn_cnt_map[x, y] = SAME
            else:
                # 其它颜色的块用 -2 表示
                for (x, y) in self.poslist[idx]:
                    turn_cnt_map[x, y] = OTHER
            # 空位置还是 -1
        

        # 每个折数的方块位置的列表
        # 这里只保存某个折数的空方块
        all_turn_intermd = []
        # 首先将初始块的位置加入列表中
        all_turn_intermd.append([pos])
        # 每个折数对应的相同的块的位置
        all_turn_target = []

        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]

        for turn_cnt in range(3):
            # 遍历当前折数的所有位置
            # 第一次循环时就只有初始块
            for curr_turn_pos in all_turn_intermd[turn_cnt]:
                new_turn_intermd = []
                new_turn_target = []

                # 尝试向 4 个方向扩展
                for dir in range(4):
                    (x, y) = curr_turn_pos
                    # 是否在棋盘内部
                    while (self.isInsideBoard((x, y))):
                        # 向指定方向移动
                        x += dx[dir]
                        y += dy[dir]
                        # 为空，可以走
                        if turn_cnt_map[x, y] == EMPTY:
                            turn_cnt_map[x, y] = turn_cnt
                            new_turn_intermd.append((x, y))
                        # 是相同的块，将其加入 turn_target 中，停止
                        elif turn_cnt_map[x, y] == SAME:
                            new_turn_target.append((x, y))
                            break
                        # 是不同的块，停止
                        elif turn_cnt_map[x, y] == OTHER:
                            break
                        # 遇到了标记过折数的的空块，不再重复标记
                        # 但是不停止，而是继续向当前方向行进
                        elif turn_cnt_map[x, y] >= 0:
                            pass

            all_turn_intermd.append(new_turn_intermd)
            all_turn_target.append(new_turn_target)

        return all_turn_target
        
        


    def actions(self, state):
        # 遍历每一种图案
        for idx in range(1, self.kinds + 1):
            for pos_start in self.poslist[idx]:
                target_lists = self.findPath(idx, pos_start, state)
                
        pass


    def move(self, action):
        pass




    def is_goal(self, state):
        return state == self.goal_node.state

    def g(self, g_cost, from_state, action, to_state):
        return g_cost + 1
    
    def h(self, this_state):
        h_cost = 0
        goal_state = self.goal_node.state
        for row1 in range(self.n):
            for col1 in range(self.n):
                num = this_state[row1][col1]
                row2 = np.where(np.array(goal_state) == num)[0][0]
                col2 = np.where(np.array(goal_state) == num)[1][0]
                h_cost += abs(row1 - row2) + abs(col1 - col2)
        return h_cost

def DFS(problem):
    pass


if __name__ == "__main__":

    problem = LinkGame(4, 6, 5)

    state = problem.generateBorad(empty=False)
    print(state)

    problem.sliceBoard(state)
    for idx in range(problem.kinds + 1):
        print("idx = ", idx)
        print(problem.poslist[idx])
    pass