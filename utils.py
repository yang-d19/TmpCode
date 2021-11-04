#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque

import sortedcontainers


class Position:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __add__(self, rhs):
        return Position(self.x + rhs.x, self.y + rhs.y)

    def __sub__(self, rhs):
        return Position(self.x - rhs.x, self.y - rhs.y)
    
    def __truediv__(self, rhs):
        return Position(self.x / rhs, self.y / rhs)

    def __floordiv__(self, rhs):
        return Position(self.x // rhs, self.y // rhs)
    
    def __repr__(self):
        return "<Position: x={}  y={}>" \
                .format(self.x, self.y)


class Queue(object):
    def __init__(self):
        self._items = deque([])

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.popleft() \
            if not self.empty() else None

    def empty(self):
        return len(self._items) == 0

    def find(self, item):
        return self._items.index(item) if item in self._items else None


class Stack(object):
    def __init__(self):
        self._items = list()

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.pop() if not self.empty() else None

    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._items)


class PriorityQueue(object):

    def __init__(self, node):
        self._queue = sortedcontainers.SortedList([node])

    def push(self, node):
        self._queue.add(node)

    def pop(self):
        return self._queue.pop(index=0)

    def empty(self):
        return len(self._queue) == 0

    def compare_and_replace(self, i, node):
        if node < self._queue[i]:
            self._queue.pop(index=i)
            self._queue.add(node)

    def find(self, node):
        try:
            loc = self._queue.index(node)
            return loc
        except ValueError:
            return None
    # def find(self, node):
    #     for i in range(len(self._queue)):
    #         if self._queue[i].state == node.state:
    #             return i
    #     return None


class Set(object):
    def __init__(self):
        self._items = set()

    def add(self, item):
        self._items.add(item)

    def remove(self, item):
        self._items.remove(item)

    def inset(self, item):
        if item in self._items:
            return True
        else:
            return False


class Dict(object):
    def __init__(self):
        self._items = dict()

    def add(self, key, value):
        self._items.update({key: value})

    def remove(self, key):
        self._items.pop(key, None)

    def find(self, key):
        return self._items[key] if key in self._items else None
