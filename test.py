import numpy as np
import math
import random
from utils import *
from heapq import *

# A = np.zeros((3, 5))
# print(A)

# for i in range(4, 9):
#     print(i)

def list2str(state):
    return tuple(np.array(state).flatten())


if __name__ == "__main__":

    state = np.matrix([[1, 2, 3], [5, 6, 7]])
    print(list2str(state))

    # a = [(1, 2), (3, 4), (5, 6)]
    # a.pop(a.index((1, 2)))
    # print(a)

    # a = PriorityQueue(1, -3, 9, 4, 6, -1, 12, 4)

    # for i in range(5):
    #     print(a.pop())

    # heap = [1, 0, 9, 13, -4, 5, 15, -6]
    # heapify(heap)

    # for i in range(4):
    #     print(heappop(heap))
    
    # print(heap)


    

    # A = np.zeros((3, 4))
    # print(A)
    # A = np.matrix((3, 4), 2)
    # print(A)
    # a = [[] for i in range(3)]
    # print(a)
#     
#     a = [1, 3 ,5, 7]
#     if 1 not in a:
#         print("okk")
#     if 2 not in a:
#         print("pop")
#     
#     pos1 = (1, 2)
#     pos2 = (4, 5)
#     t = 3
#     x = ((pos1, pos2), t)
#     (((x1, y1), (x2, y2)), turn) = x
# #    print(p1, p2, turn)
#     print(x1, y1, x2, y2, turn)

    # x = np.matrix([[1, 2, 3], [2, 5, 7]])
    # y = np.matrix([[1, 2, 3], [2, 5, 7]])
    # z = np.matrix([[1, 2, 3], [2, 5, 8]])
    # if (x == y).all():
    #     print("x y")
    # if (x == z).any():
    #     print("x z")
    




