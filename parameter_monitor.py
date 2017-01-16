from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import sys
from argparse import ArgumentParser
import time
import math

# Queue Delay, Throughput, Reward, Drop Rate

names = ['Queue Delay', 'Drop Rate', 'Throughput', 'Reward']
total_win = len(names)

average_kernel = 20
fs = 0.02 * average_kernel

win = pg.GraphicsWindow(title="Parameter Monitor")

plots = []
curve_plots = []
curve_y = []
win_size = int(200 / fs) # 300 sec
curve_x = np.arange(-win_size, 0) * fs

for i in range(len(names)):
    p = win.addPlot(title="%s Monitor" % names[i], row=i, col=0)
    plots.append(p)
    p.enableAutoRange('y', 1)
    curve_plots.append(p.plot(pen=pg.mkPen('y', width=2)))
    curve_y.append(np.zeros(win_size))

bar_queue_delay = np.ones(win_size) * 50
bar_plot = plots[0].plot(pen=pg.mkPen('r', width=1))

t_start = time.time()
t_now = time.time()

while True:
    global curve_plots, curve_y, t_now, win_size, names
    start_time = time.time()

    values = np.zeros((average_kernel, total_win), dtype=np.float64)
    for i in range(average_kernel):
        #line = '%.f,%f,%f' % (math.sin(t_now), 0.01 * (t_now - t_start) + math.sin(2 * t_now), -0.01 * (t_now - t_start) + math.cos(2 * t_now))

        line = sys.stdin.readline()
        new_values = line.split(',')
        assert len(new_values) == len(names)
        for j in range(total_win):
            values[i, j] = float(new_values[j])

    average_value = np.average(values, axis=0)
    if average_value[0] > 300:
        average_value[0] = 300
    t_now = time.time()
    for i in range(len(names)):
        curve_y[i][:-1] = curve_y[i][1:]
        curve_y[i][-1] = average_value[i]
        curve_plots[i].setData((curve_x + t_now - t_start), curve_y[i])

    bar_plot.setData((curve_x + t_now - t_start), bar_queue_delay)
    pg.QtGui.QApplication.processEvents()
    # print("Prepare Experiment: Time elaspsed: {:.3f} s".format(time.time() - start_time))


if __name__ == '__main__':
    QtGui.QApplication.instance().exec_()
