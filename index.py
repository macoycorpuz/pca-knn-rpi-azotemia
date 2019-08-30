import Adafruit_ADS1x15
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import pca, knn

## Draw Figures
fig = plt.figure()
fig.canvas.manager.full_screen_toggle()
ax_graph = fig.add_axes([0.13, 0.25, 0.80, 0.68])
ax_air = plt.axes([0.03, 0.04, 0.13, 0.05])
ax_healthy = plt.axes([0.16, 0.04, 0.13, 0.05])
ax_azotemic = plt.axes([0.29, 0.04, 0.13, 0.05])
ax_pca_btn = plt.axes([0.44, 0.04, 0.13, 0.05])
ax_knn_btn = plt.axes([0.57, 0.04, 0.13, 0.05])
ax_close = plt.axes([0.85, 0.04, 0.13, 0.05])
ax_text = plt.text(-1, 18, '', fontsize=9, fontweight='bold')
btnAir = Button(ax_air, 'None')
btnHealthy = Button(ax_healthy, 'Healthy')
btnAzotemic = Button(ax_azotemic, 'Azotemic')
btnPCA = Button(ax_pca_btn, 'PCA')
btnKNN = Button(ax_knn_btn, 'KNN')
btnExit = Button(ax_close, 'Exit')
btnAir.label.set_fontsize(9)
btnHealthy.label.set_fontsize(9)
btnAzotemic.label.set_fontsize(9)
btnPCA.label.set_fontsize(9)
btnKNN.label.set_fontsize(9)
btnExit.label.set_fontsize(9)

## Initialize ADC
x = []
target = 'air'
predict = False
ctr = 0
sensors = {'MQ2':[], 'MQ3':[], 'MQ4':[], 'MQ6':[], 'MQ7':[], 'MQ8':[], 'MQ135':[]}
colors = {'MQ2': 'b', 'MQ3': 'g', 'MQ4': 'r', 'MQ6': 'c', 'MQ7': 'm', 'MQ8':'y', 'MQ135': 'k'}
adc1 = Adafruit_ADS1x15.ADS1115(address=0x49)
adc2 = Adafruit_ADS1x15.ADS1115(address=0x48)

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

def onClickKNN(e):
    global predict, ctr
    clear_test_data()
    ctr = 0
    predict = True
    
def onClickExit(e):
    plt.close()

def design_rtv_graph(ax_graph):
    ax_graph.set_title('Real Time View', fontsize=10, fontweight="bold", loc="left")
    ax_graph.set_xlabel('Time (s)', fontsize=8, fontweight="bold")
    ax_graph.set_ylabel('MQ sensor values (V)', fontsize=8, fontweight="bold")

def save_data(data): 
    global target
    with open('_history.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(data)

    if predict:
        with open('_test.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(data)

    if target != 'air':
        with open('_dataset.csv', 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(data + [target])

def clear_test_data():
    with open('_test.csv', 'w+') as fd:
        writer = csv.writer(fd)
        writer.writerow(list(sensors.keys()))

def animate_rtv(i, x, sensors, colors, start_time):
    ax_graph.clear()

    data = []
    x.append(round(time.time()-start_time, 5))
    for i, (sensor, values) in enumerate(sensors.items()):
        adc = adc1.read_adc(i) if i < 4 else adc2.read_adc(i-4)
        volts = round(adc * (4.096/32767), 5)
        values.append(volts)
        data.append(volts)
        
        lbl = "{}: {}".format(sensor, str(values[-1]))
        ax_graph.plot(x, values, color=colors[sensor], label=lbl)  
        ax_graph.legend(loc='upper left', borderaxespad=0.5, prop={'size': 7})
    design_rtv_graph(ax_graph)
    save_data(data)

    global predict, ctr
    if predict:
        ctr += 1
        ax_text.set_text(str(ctr) + '%')
        if ctr >= 100:
            ax_text.set_text(knn.knn())
    else:
        ax_text.set_text('')


btnAir.on_clicked(onClickNone)
btnHealthy.on_clicked(onClickHealthy)
btnAzotemic.on_clicked(onClickAzotemic)
btnPCA.on_clicked(onClickPCA)
btnKNN.on_clicked(onClickKNN)
btnExit.on_clicked(onClickExit)
ani = animation.FuncAnimation(fig, animate_rtv,frames = 10, interval=1000, fargs=(x, sensors, colors, time.time()))
plt.show()
