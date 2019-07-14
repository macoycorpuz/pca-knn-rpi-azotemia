import time
import Adafruit_ADS1x15
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#* Initialize ADC and Sensors
adc_6_to_135 = Adafruit_ADS1x15.ADS1115(address=0x48, busnum=1)
adc_2_to_4 = Adafruit_ADS1x15.ADS1115(address=0x49, busnum=1)
sensors = ['MQ2', 'MQ3', 'MQ4', 'MQ6', 'MQ7', 'MQ8', 'MQ135']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

#*Initialize Figure
fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()
fig.suptitle('Real time view',fontsize=20)
plt.ylabel('MQ Values (V)',fontsize=14)
plt.xlabel('Iterations',fontsize=12)
plt.xticks(rotation=45)
x, y = [], {}

#* Print Sensors Header
# print('| {0:>7} | {1:>7} | {2:>7} | {3:>7} | {4:>7} | {5:>7} | {6:>7} | {7:>7} |'.format(*sensors))
# print('-' * 74)
# print('\n' * 3)

#* Read Sensor Values
# start = time.time()
ctr = 1
while True:
    values = [0]*8

    x.append(ctr)
    for i in range(7):
        adc = adc_2_to_4.read_adc(i) if i < 4 else adc_6_to_135.read_adc(i-4)
        volts = adc * (4.096/32767)
        values[i] = round(volts, 5)
        sensor = sensors[i]
        y.setdefault(sensor, []).append(values[i])
        ax.plot(x, y[sensor], color=colors[i])

    fig.canvas.draw()
    plt.pause(.2)


    #* Print Sensor Values and current iteration in the Terminal
    # print("\033[F"*5)
    # print('| {0:>7} | {1:>7} | {2:>7} | {3:>7} | {4:>7} | {5:>7} | {6:>7} |\n'.format(*values))
    # print('Iteration: {}'.format(ctr))
    # print('Elapsed Time: {0:.2f}'.format(time.time()-start))
    # time.sleep(0.1)
    # ctr += 1