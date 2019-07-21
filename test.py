####################
## Main test file ##
####################
 
import Adafruit_ADS1x15
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## Initialize Analog to digital converter
# adc1 = MQ2 to MQ4
# adc2 = MQ6 to MQ135
adc1 = Adafruit_ADS1x15.ADS1115(address=0x49)
adc2 = Adafruit_ADS1x15.ADS1115(address=0x48)
fig = plt.figure(figsize=(10,9))
# ax_pca = fig.add_axes([0.6, 0.1, 0.35, 0.35])
ax_rtv = fig.add_axes([0.1, 0.15, 0.65, 0.7])

def design_pca_graph(ax_pca):
    ax_pca.set_title('2 Component PCA', fontsize=18, fontweight="bold")
    ax_pca.set_xlabel('pca1', fontsize=12, fontweight="bold")
    ax_pca.set_ylabel('pca2', fontsize=12, fontweight="bold")

def design_rtv_graph(ax_rtv):
    ax_rtv.set_title('Real Time View', fontsize=18, fontweight="bold", loc="left")
    ax_rtv.set_xlabel('Time', fontsize=12, fontweight="bold")
    ax_rtv.set_ylabel('MQ sensor values', fontsize=12, fontweight="bold")

def animate_rtv(i, x, sensors, colors, start_time):
    ax_rtv.clear()
    x.append(round(time.time()-start_time, 5))
    for i, (sensor, values) in enumerate(sensors.items()):
        adc = adc1.read_adc(i) if i < 4 else adc2.read_adc(i-4)
        volts = round(adc * (4.096/32767), 5)
        values.append(volts)
        
        lbl = "{}: {}".format(sensor, str(values[-1]))
        ax_rtv.plot(x, values, color=colors[sensor], label=lbl)  
        ax_rtv.legend(bbox_to_anchor=(1.05, 1.02), loc='upper left', borderaxespad=0.5)
    
    design_rtv_graph(ax_rtv)
  
def save_data(data_name, sensors): 
    with open(name + '.csv', mode='w') as csv_file:
        csv_file = csv.writer(csv_file, delimiter=',')
    
  
x = []
sensors = {'MQ2':[], 'MQ3':[], 'MQ4':[], 'MQ6':[], 'MQ7':[], 'MQ8':[], 'MQ135':[]}
colors = {'MQ2': 'b', 'MQ3': 'g', 'MQ4': 'r', 'MQ6': 'c', 'MQ7': 'm', 'MQ8':'y', 'MQ135': 'k'}

# design_pca_graph(ax_pca)
ani = animation.FuncAnimation(fig, animate_rtv, interval=1000, fargs=(x, sensors, colors, time.time()))
plt.show()
