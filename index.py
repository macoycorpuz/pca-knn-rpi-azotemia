import Adafruit_ADS1x15
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import pca

adc1 = Adafruit_ADS1x15.ADS1115(address=0x49)
adc2 = Adafruit_ADS1x15.ADS1115(address=0x48)
fig = plt.figure(figsize=(10,9))
fig.canvas.manager.full_screen_toggle()

ax_rtv = fig.add_axes([0.1, 0.15, 0.65, 0.7])
ax_air = plt.axes([0.10, 0.03, 0.1, 0.05])
ax_healthy = plt.axes([0.20, 0.03, 0.1, 0.05])
ax_azotemic = plt.axes([0.30, 0.03, 0.1, 0.05])
ax_pca_btn = plt.axes([0.41, 0.03, 0.1, 0.05])

btnAir = Button(ax_air, 'None')
btnHealthy = Button(ax_healthy, 'Healthy')
btnAzotemic = Button(ax_azotemic, 'Azotemic')
btnPCA = Button(ax_pca_btn, 'Show PCA')

x = []
target = 'air'
sensors = {'MQ2':[], 'MQ3':[], 'MQ4':[], 'MQ6':[], 'MQ7':[], 'MQ8':[], 'MQ135':[]}
colors = {'MQ2': 'b', 'MQ3': 'g', 'MQ4': 'r', 'MQ6': 'c', 'MQ7': 'm', 'MQ8':'y', 'MQ135': 'k'}

def design_rtv_graph(ax_rtv):
    ax_rtv.set_title('Real Time View', fontsize=18, fontweight="bold", loc="left")
    ax_rtv.set_xlabel('Time', fontsize=12, fontweight="bold")
    ax_rtv.set_ylabel('MQ sensor values', fontsize=12, fontweight="bold")

def onClickPCA(e):
    pca.pca()

def onClickNone(e):
    global target
    target = 'air'

def onClickHealthy(e):
    global target
    target = 'healthy'

def onClickAzotemic(e):
    global target
    target = 'azotemic'

def save_data(values): 
    with open('history.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(values)

    with open('dataset.csv', 'a') as fd2:
        writer = csv.writer(fd2)
        writer.writerow(values + [target])

def animate_rtv(i, x, sensors, colors, start_time):
    data = []

    ax_rtv.clear()
    x.append(round(time.time()-start_time, 5))
    for i, (sensor, values) in enumerate(sensors.items()):
        adc = adc1.read_adc(i) if i < 4 else adc2.read_adc(i-4)
        volts = round(adc * (4.096/32767), 5)
        values.append(volts)
        data.append(volts)
        
        lbl = "{}: {}".format(sensor, str(values[-1]))
        ax_rtv.plot(x, values, color=colors[sensor], label=lbl)  
        ax_rtv.legend(bbox_to_anchor=(1.05, 1.02), loc='upper left', borderaxespad=0.5)
    
    save_data(data)
    design_rtv_graph(ax_rtv)

btnAir.on_clicked(onClickNone)
btnHealthy.on_clicked(onClickHealthy)
btnAzotemic.on_clicked(onClickAzotemic)
btnPCA.on_clicked(onClickPCA)
ani = animation.FuncAnimation(fig, animate_rtv,frames = 10, interval=1000, fargs=(x, sensors, colors, time.time()))
plt.show()
