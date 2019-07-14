import time
import Adafruit_ADS1x15
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# adc1 = MQ2 to MQ4
# adc2 = MQ6 to MQ135
class Sensors:

    sensors = ['MQ2', 'MQ3', 'MQ4', 'MQ6', 'MQ7', 'MQ8', 'MQ135']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    def __init__(self):
        self.adc1 = Adafruit_ADS1x15.ADS1115(address=0x49)
        self.adc2 = Adafruit_ADS1x15.ADS1115(address=0x48)
    
    def print_cli_header(self):
        print('| {0:>7} | {1:>7} | {2:>7} | {3:>7} | {4:>7} | {5:>7} | {6:>7} | {7:>7} |'.format(*sensors))
        print('-' * 74)
        print('\n' * 3)

    def print_cli(self):
        print("\033[F"*5)
        print('| {0:>7} | {1:>7} | {2:>7} | {3:>7} | {4:>7} | {5:>7} | {6:>7} |\n'.format(*values))
        print('Iteration: {}'.format(ctr))
        print('Elapsed Time: {0:.2f}'.format(time.time()-start))
        time.sleep(0.1)
        ctr += 1

class GUI:

    def __init__(self):
        self.ui = plt.subplots()

    def draw_real_time_figure(self):
        self.fig, self.ax = plt.subplots()  
        self.fig.show()  
        self.fig.suptitle('Real time view',fontsize=20)
        plt.ylabel('MQ Values (V)',fontsize=14)
        plt.xlabel('Iterations',fontsize=12)
        plt.xticks(rotation=45)

if __name__ == '__main__':  
    ui = GUI()


    # x, y = [], {}
    # ctr = 1
    # while True:
    #     values = [0]*8

    #     x.append(ctr)
    #     for i in range(7):
    #         adc = adc1.read_adc(i) if i < 4 else adc2.read_adc(i-4)
    #         volts = adc * (4.096/32767)
    #         values[i] = round(volts, 5)
    #         sensor = sensors[i]
    #         y.setdefault(sensor, []).append(values[i])
    #         ax.plot(x, y[sensor], color=colors[i])

    #     fig.canvas.draw()
    #     plt.pause(.2)