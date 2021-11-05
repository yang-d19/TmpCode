#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import QTimer,QDateTime
import sys
import numpy as np

# start_x = 100
# start_y = 100
# block_size = 100

class PaintSpace(QWidget):
    
    def __init__(self, init_state, actions=[]):

        super().__init__()
        # 类型: np.matrix
        self.state = init_state
        # 每次消除的两个块的坐标的列表
        self.block_pairs = actions
        # 当前绘制到了第几步
        self.cnt = 0
        # 绘制的方块大小
        self.block_size = 100
        # 方块与方块之间的间隙
        self.margin = 20
        # 边角的方块距离边界的距离
        self.dist = 100
        # 目前仅支持最大 9 种颜色
        self.colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200), 
                        (100, 100, 0), (100, 0, 100), (0, 100, 100),
                        (100, 50, 50), (50, 100, 50), (50, 50, 100)]

        # 假定暂时不绘制消除时的连接线
        # block_pairs 中保存的是 每次消除的两点的坐标

        # 每秒刷新一次绘图区域
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refreshState)
        self.timer.start(1500)

        self.initUI()


    def refreshState(self):

        # 所有绘制操作已经执行完，关闭计时器，直接返回
        if self.cnt >= len(self.block_pairs):
            self.timer.stop()
            return

        ((x1, y1), (x2, y2)) = self.block_pairs[self.cnt]
        self.state[x1, y1] = 0
        self.state[x2, y2] = 0
        self.repaint()
        # 切换到下一次操作
        self.cnt += 1

        
    def initUI(self):

        # 窗口距离显示屏左上角的距离
        figureDist = 500

        rowCnt = self.state.shape[0]
        colCnt = self.state.shape[1]
        # 窗口的大小
        figureWidth = 2 * self.dist + colCnt * self.block_size \
                    + (colCnt - 1) * self.margin
        figureHeight = 2 * self.dist + rowCnt * self.block_size \
                    + (rowCnt - 1) * self.margin

        self.setGeometry(figureDist, figureDist, figureWidth, figureHeight)
        self.setWindowTitle('LinkGame')
        self.show()


    def paintEvent(self, event):

        qp = QPainter()

        qp.begin(self)

        for row in range(self.state.shape[0]):
            for col in range(self.state.shape[1]):
                block_type = self.state[row, col]
                # 当前位置为空，不绘制矩形
                if block_type == 0:
                    continue
                # 注意：绘图时的 x 是横向的，y 是纵向的
                startX = self.dist + col * (self.block_size + self.margin)
                startY = self.dist + row * (self.block_size + self.margin)
                # 获取当前块对应的颜色
                # 因为 block_type = 0 表示空，所以减 1 取颜色
                (r, g, b) = self.colors[block_type - 1]
                block_color = QColor(r, g, b)
                # 调用绘制矩形的函数
                self.drawRectangles(qp, startX, startY, self.block_size, block_color)

        qp.end()

        
    def drawRectangles(self, qp, x, y, size, block_color):
      
        qp.setPen(QColor(200, 200, 200))

        qp.setBrush(block_color)
        qp.drawRect(x, y, size, size)


def Display(init_state, actions=[]):

    app = QApplication(sys.argv)
    pSpace =  PaintSpace(init_state, actions)
    sys.exit(app.exec_())

        
if __name__ == '__main__':

    init_state = np.matrix([[0, 0, 0, 0, 0],
                            [0, 1, 2, 1, 0],
                            [0, 3, 1, 1, 0],
                            [0, 2, 0, 3, 0],
                            [0, 0, 0, 0, 0]])

    actions = [((1, 1), (1, 3)),
               ((2, 2), (2, 3)),
               ((2, 1), (3, 3)),
               ((1, 2), (3, 1))]
    
    
    Display(init_state, actions)