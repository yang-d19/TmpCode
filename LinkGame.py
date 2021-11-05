#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy

from numpy.random import shuffle
from utils import *
import math
import random
from Display import *



class Node(object):  # Represents a node in a search tree
    def __init__(self, state, parent=None, action=None, g_cost=0, h_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g_cost = g_cost  # true cost
        self.h_cost = h_cost  # heuristic score
        self.path_cost = g_cost + h_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def child_node(self, problem, action, move_cost):
        next_state = problem.move(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.g(self.g_cost, move_cost),
                         problem.h(self.state)
                         )
        return next_node


    def path(self):
        # Returns list of nodes from this node to the root node
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
        # Given the current state, return valid actions.
        # :param state:
        # :return: valid actions
        pass

    def move(self, state, action):
        pass

    def is_goal(self, state):
        pass

    def g(self, prev_cost, move_cost):
        pass
    
    def h(self, state):
        pass

    def solution(self, goal):
        # Returns actions from this node to the root node
        if goal.state is None:
            return None
        return [node.action for node in goal.path()[1:]]

    def expand_childnodes(self, node: Node): 
        # Returns a list of child nodes
        child_nodes = []

        all_turn_pairs = self.actions(node.state)

        for turn in range(len(all_turn_pairs)):
            curr_turn_pairs = all_turn_pairs[turn]
            for action in curr_turn_pairs:
                child_nodes.append(node.child_node(self, action, turn))

        # return [node.child_node(self, action) for action in self.actions(node.state)]
    

class LinkGame(Problem):

    def __init__(self, height, width, kinds):
        self.height = height
        self.width = width
        self.kinds = kinds
        # generate random initial board
        init_state = self.generateBorad(empty=False)
        # generate empty ultimate board
        goal_state = self.generateBorad()
        # contain positions of all different blocks
        self.poslist = list()
        # initial poslist by function sliceBoard
        self.sliceBoard(init_state)
        # 最大转折数
        self.max_turns = 2

        super().__init__(init_state, goal_state)


    def generateBorad(self, empty=True):
        # 游戏棋盘
        board = np.zeros((self.height, self.width), dtype=int)
        # 在核心棋盘的四周添加了空白的边，方便搜索路径
        full_board = np.zeros((self.height + 2, self.width + 2), dtype=int)
        # 生成空棋盘
        if empty == True:
            return full_board

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
    

    def sliceBoard(self, state):
        # 0号即空位置
        # 寻找第 idx 号图案的位置
        # do not find zero element
        self.poslist.append([])
        for idx in range(1, self.kinds + 1):
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
    

    # judge if one step in given direction is still in the board
    def isInsideBoard(self, pos, dir):
        UP    = 0
        RIGHT = 1
        DOWN  = 2
        LEFT  = 3

        (x, y) = pos
        h = self.height + 2
        w = self.width + 2

        if dir == UP:
            return x > 0
        elif dir == RIGHT:
            return y < w - 1
        elif dir == DOWN:
            return x < h - 1
        elif dir == LEFT:
            return y > 0


    # 找当前给定的块到相同图案的其它块的折数
    # 对于普通规则，折数最大为2
    # 对于扩展规则，折数无最大限制
    def findPath(self, index, pos, state):
        # 扩展过的棋盘大小
        h = state.shape[0]
        w = state.shape[1]
        # 存储转折数的矩阵
        turn_cnt_map = np.zeros((h, w), dtype=int)
        # 由于我的设置是 -1 表示 EMPTY
        # 所以所有 map 所有元素减去 1，这样初始就是全空棋盘
        turn_cnt_map -= 1

        # const variable
        # 特殊块都是小于 0 的
        EMPTY  = -1  # 空
        OTHER  = -2  # 不同的块
        SAME   = -3  # 同图案的块
        SELF   = -4  # 初始块
        FINDED = -5  # 已经被找到的块
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
        # this place is the start block
        turn_cnt_map[pos[0], pos[1]] = SELF

        # 每个折数的方块位置的列表
        # 这里只保存某个折数的空方块
        all_turn_intermd = []
        # 每个折数对应的相同的块的位置
        all_turn_target = []

        # 分别是上、下、左、右四个方向
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]

        # turn_cnt 从 0 到 2
        for turn_cnt in range(3):
            # 遍历当前折数的所有位置

            # 转折数为 0 的一定是从初始块开始的
            if turn_cnt == 0:
                pre_turn_intermd = [pos]
            else:
                # 转折数为 n 的块一定是由转折数为 n-1 的块走直线得到的
                pre_turn_intermd = all_turn_intermd[turn_cnt - 1]
            
            # 从当前转折数的块上走出来的转折数加一的块的位置
            new_turn_intermd = []
            new_turn_target = []

            for curr_turn_pos in pre_turn_intermd:

                # 尝试向 4 个方向扩展
                for dir in range(4):
                    (x, y) = curr_turn_pos
                    # can move in this direction or not
                    while (self.isInsideBoard((x, y), dir)):
                        # 向指定方向移动
                        x += dx[dir]
                        y += dy[dir]
                        # 为空，在 turn_cnt_map 图上留下标记
                        # 将当前位置加入 new_turn_intermd，
                        # 继续尝试向该方向走
                        if turn_cnt_map[x, y] == EMPTY:
                            turn_cnt_map[x, y] = turn_cnt
                            new_turn_intermd.append((x, y))
                        # 是相同的块，将其加入 turn_target 中，
                        # turn_cnt_map 标记该块为已访问
                        # 停止
                        elif turn_cnt_map[x, y] == SAME:
                            # print("add SAME block, pos = ", x, y)
                            # print("turn_cnt = ", turn_cnt)
                            turn_cnt_map[x, y] = FINDED
                            new_turn_target.append((x, y))
                            break
                        # 是不同的块，或是初始块，或是已经访问到的块
                        # 停止该方向的前进
                        elif turn_cnt_map[x, y] == OTHER \
                            or turn_cnt_map[x, y] == SELF:
                            break
                        # 遇到了标记过折数的的空块，不再重复标记
                        # 但是不停止，而是继续向当前方向行进
                        elif turn_cnt_map[x, y] >= 0:
                            pass

            all_turn_intermd.append(new_turn_intermd)
            all_turn_target.append(new_turn_target)

            # print("turn_cnt = ", turn_cnt)
            # print("\nintermedium: \n", new_turn_intermd)
            # print("\ntargets: \n", new_turn_target)
            # print("\nturn cnt map: \n", turn_cnt_map)
        return all_turn_target


    def actions(self, state):
        max_turns = 2
        # each dimension of this list contains pairs of two block
        # that are able to eliminate with each other
        all_turn_pairs = [[] for i in range(max_turns + 1)]

        # 遍历每一种图案
        for idx in range(1, self.kinds + 1):
            # 对于该种图案的每个位置
            for start_pos in self.poslist[idx]:
                # 找到所有可以和当前块消的位置
                all_turn_target = self.findPath(idx, start_pos, state)
                # 遍历每一个折数
                for turn in range(len(all_turn_target)):
                    curr_turn_target = all_turn_target[turn]
                    # 遍历该折数对应的所有终止点
                    for end_pos in curr_turn_target:
                        # 将待消两块组合起来
                        pos_pair = (start_pos, end_pos)
                        rev_pos_pair = (end_pos, start_pos)
                        # 如果与其对位的那个点没有先把...
                        # if this pair has not been put into pairList
                        if rev_pos_pair not in all_turn_pairs[turn]:
                            # 加入到可执行动作列表中
                            all_turn_pairs[turn].append(pos_pair)
                
        return all_turn_pairs


    def move(self, state, action):

        (((x1, y1), (x2, y2))) = action

        assert state[x1, y1] == state[x2, y2], "消除的两个方块不是同一种类"
        block_type = state[x1, y1]
        # delete element in poslist
        self.poslist[block_type].pop(self.poslist[block_type].index((x1, y1)))
        self.poslist[block_type].pop(self.poslist[block_type].index((x2, y2)))
        # change state
        new_state = deepcopy(state)
        # the two positions are now empty
        new_state[x1, y1] = 0
        new_state[x2, y2] = 0

        return new_state
        

    def is_goal(self, state):
        return (state == self.goal_node.state).all()

    def g(self, pre_cost, move_cost):
        # move cost 即为转折数，为防止路径代价不增
        # 当前步代价视为转折数加一
        return pre_cost + move_cost + 1
    
    def h(self, state):
        # 不使用启发式函数 h 时，A*算法就是一致代价搜索
        h_cost = 0
        # goal_state = self.goal_node.state

        # for idx in range(1, self.kinds + 1):
            
        return h_cost


def DFS(problem: LinkGame, state, path):
    # completed, all blocks have been eliminated
    # now the board is empty
    if problem.is_goal(state):
        return True

    all_turn_pairs = problem.actions(state)

    for turn in range(len(all_turn_pairs)):

        block_pairs = all_turn_pairs[turn]

        for action in block_pairs:
            # 因为 move 函数中使用了 deepcopy 深拷贝
            # 当前的 new_state 和旧的 state 互不干扰
            new_state = problem.move(state, action)
            # add temp action to path
            path.append(action)
            # 递归寻找下一对可以消除的方块
            res = DFS(problem, new_state, path)
            # if already found a way to eliminate all blocks
            # than do not try other ways
            if res == True:
                return True
            else:
                # 去除不可行的行动
                # new_state 和 state 独立，不要变
                path.pop(-1)

    return False




if __name__ == "__main__":

    problem = LinkGame(4, 6, 5)

    # ----- DFS -----    
    # solution_path = []
    # res = DFS(problem, problem.init_node.state, solution_path)
    # if res == True:
    #     print("Solution found")
    #     Display(problem.init_node.state, solution_path)
    # else:
    #     print("Can't find a way to eliminate all blocks")
    # ----- END -----

    
    
    


    # init_state = problem.init_node.state
    


    # ----- DEBUG -----
    # print(problem.init_node.state, "\n")
    # for ele in problem.poslist:
    #     print(ele)
    # print("")
    # ----- END -----
    
    # all_turn_target = problem.findPath(1, problem.poslist[1][0], init_state)

    # block_pairs = problem.actions(init_state)




    # ----- DEBUG -----
    # for turn in range(3):
    #     print(block_pairs[turn])
    # ----- END -----

    


    # ----- DEBUG -----
    # print("")
    # for turn in range(len(all_turn_target)):
    #     print(all_turn_target[turn])
    # ------ END -----


    # ----- DEBUG -----
    # state = problem.generateBorad(empty=False)
    # print(state)
    # problem.sliceBoard(state)
    # for idx in range(problem.kinds + 1):
    #     print("idx = ", idx)
    #     print(problem.poslist[idx])
    # pass
    # ------ END -----
