import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GUI:

    def __init__(self):
        print('Initialized GUI.')

    def draw_real_time_figure(self):
        self.fig, self.ax = plt.subplots(1, 2)  
        self.fig.show()  
        self.fig.suptitle('Real time view',fontsize=20)
        plt.ylabel('MQ Values (V)',fontsize=14)
        plt.xlabel('Iterations',fontsize=12)
        plt.xticks(rotation=45)
        plt.show()

if __name__ == '__main__':
    ui = GUI()
    ui.draw_real_time_figure()

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i, filename):
    graph_data = open(filename, 'r').read()
    lines = graph_data.split('\n')
    xr, yr = [], []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xr.append(x)
            yr.append(y)
    ax1.clear()
    ax1.plot(xr, yr)

ani = animation.FuncAnimation(fig, animate, fargs=("database.csv",))
plt.show()