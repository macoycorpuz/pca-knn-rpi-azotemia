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
fig = plt.figure()
fig.canvas.manager.full_screen_toggle()

ax_rtv = fig.add_axes([0.11, 0.25, 0.61, 0.68])
ax_air = plt.axes([0.03, 0.04, 0.13, 0.05])
ax_healthy = plt.axes([0.16, 0.04, 0.13, 0.05])
ax_azotemic = plt.axes([0.29, 0.04, 0.13, 0.05])
ax_pca_btn = plt.axes([0.44, 0.04, 0.13, 0.05])
ax_knn_btn = plt.axes([0.57, 0.04, 0.13, 0.05])
ax_svm_btn = plt.axes([0.70, 0.04, 0.13, 0.05])
ax_close = plt.axes([0.85, 0.04, 0.13, 0.05])

btnAir = Button(ax_air, 'None')
btnHealthy = Button(ax_healthy, 'Healthy')
btnAzotemic = Button(ax_azotemic, 'Azotemic')
btnPCA = Button(ax_pca_btn, 'PCA')
btnKNN = Button(ax_knn_btn, 'KNN')
btnSVM = Button(ax_svm_btn, 'SVM')
btnExit = Button(ax_close, 'Exit')

btnAir.label.set_fontsize(9)
btnHealthy.label.set_fontsize(9)
btnAzotemic.label.set_fontsize(9)
btnPCA.label.set_fontsize(9)
btnKNN.label.set_fontsize(9)
btnSVM.label.set_fontsize(9)
btnExit.label.set_fontsize(9)

x = []
target = 'air'
sensors = {'MQ2':[], 'MQ3':[], 'MQ4':[], 'MQ6':[], 'MQ7':[], 'MQ8':[], 'MQ135':[]}
colors = {'MQ2': 'b', 'MQ3': 'g', 'MQ4': 'r', 'MQ6': 'c', 'MQ7': 'm', 'MQ8':'y', 'MQ135': 'k'}

def design_rtv_graph(ax_rtv):
    ax_rtv.set_title('Real Time View', fontsize=10, fontweight="bold", loc="left")
    ax_rtv.set_xlabel('Time (s)', fontsize=8, fontweight="bold")
    ax_rtv.set_ylabel('MQ sensor values (V)', fontsize=8, fontweight="bold")

def onClickNone(e):
    global target
    target = 'air'

def onClickHealthy(e):
    global target
    target = 'healthy'

def onClickAzotemic(e):
    global target
    target = 'azotemic'

def onClickPCA(e):
    pca.pca()

def onClickExit(e):
    plt.close()

def save_data(data): 
    global target
    with open('history.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(data)

    if target != 'air':
        with open('dataset.csv', 'a') as fd2:
            writer = csv.writer(fd2)
            writer.writerow(data + [target])

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
        ax_rtv.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left', borderaxespad=0.5, prop={'size': 7})
    
    save_data(data)
    design_rtv_graph(ax_rtv)

btnAir.on_clicked(onClickNone)
btnHealthy.on_clicked(onClickHealthy)
btnAzotemic.on_clicked(onClickAzotemic)
btnPCA.on_clicked(onClickPCA)
btnExit.on_clicked(onClickExit)
ani = animation.FuncAnimation(fig, animate_rtv,frames = 10, interval=1000, fargs=(x, sensors, colors, time.time()))
plt.show()
