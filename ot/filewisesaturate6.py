##########################################################
#Python file for main article
#Read.Save,Plot
#filewisesaturate with all sensors without gui
#directed to Comm6 folder
##########################################################
#4/11/2018

import time
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Adafruit_ADS1x15
import Adafruit_DHT
import csv
import sys
import I2C_LCD_driver
import os
import os.path
import cPickle as pickle
import ast
from pathlib import Path


        
def training_read():
    #iterations
    reading_cycle = input("How Many Iterations?: ")
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Iterations:", 1)
    mylcd.lcd_display_string("%s" %(reading_cycle), 2)
    
    #actual_output
    actual_output = input("Actual Output (Precursor =1 , Non-Precursor = 0: ")
    prec = ' '
    if actual_output == 0:
        prec='Non-Precursor'
    elif actual_output == 1:
        prec='Precursor'
        
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Actual Output", 1)
    mylcd.lcd_display_string("%s" %(prec), 2)
    
    file_check_loop = 0
    while file_check_loop == 0:
        classify_output = raw_input("Name of Class: ")
        
        mylcd.lcd_clear()
        mylcd.lcd_display_string("Class Name:", 1)
        mylcd.lcd_display_string("%s" %(classify_output), 2)
        
        class_to_file = 'database'+classify_output+'.txt'
        My_file = Path('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm6/'+class_to_file)
        if My_file.is_file():
            print('Already have a Class named: '+classify_output)
            
            mylcd.lcd_clear()
            mylcd.lcd_display_string("Already Exist", 1)
            mylcd.lcd_display_string("Class: %s" %(classify_output), 2)
    
            file_check_loop = 0
        else:
            print('Created a Database named: '+class_to_file)
            file_check_loop = 1
    
    x=0
    starttime = time.time()
    while x < reading_cycle:
	# Read all the ADC channel values in a list.
	
	#formula is Sensor Voltage = AnalogReading / 4.096V / 32767 if gain is 1
	#---------------------------------------------------
        O2V = ((adc.read_adc(0, gain=GAIN))*4.096/32767)
        MQ8V = ((adc.read_adc(1, gain=GAIN))*4.096/32767)
        MQ2V = ((adc.read_adc(2, gain=GAIN))*4.096/32767)
        MQ138V = ((adc.read_adc(3, gain=GAIN))*4.096/32767)
        CO2V= ((adc2.read_adc(0, gain=GAIN))*4.096/32767)
	MQ136V = ((adc2.read_adc(1, gain=GAIN))*4.096/32767)
        MQ7V = ((adc2.read_adc(2, gain=GAIN))*4.096/32767)
        MQ4V = ((adc2.read_adc(3, gain=GAIN))*4.096/32767)
        
        My_data = open('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm6/'+class_to_file,"a+")
        My_data.write(str(O2V) +",")
        My_data.write(str(MQ8V) +",")
        My_data.write(str(MQ2V) +",")
        My_data.write(str(MQ138V) +",")    
        My_data.write(str(CO2V) +",")
	My_data.write(str(MQ136V) +",")   
        My_data.write(str(MQ7V) +",")
        My_data.write(str(MQ4V) +",")
        My_data.write(str(classify_output) +",")
        My_data.write(str(actual_output) +"\n")
    
        sys.stdout.write("\r")
        sys.stdout.write("\033[K")
        sys.stdout.write("MQ8: %g V, MQ138: %g V, CO2: %g V, O2: %g V, MQ2: %g V, MQ136: %g V, MQ7: %g V, MQ4: %g V " % (MQ8V, MQ138V, CO2V, O2V, MQ2V, MQ136V, MQ7V, MQ4V))        #sys.stdout.write("")
        #sys.stdout.write("Temp: %g C, Humidity: %g " % (temperature, humidity))
        
        
        My_data.close()
        sys.stdout.flush()
        time.sleep(0.176)
        x=x+1
        percent_cycle=(x/(float(reading_cycle)))*100
        mylcd.lcd_clear()
        mylcd.lcd_display_string("Class: %s" %(classify_output), 1)
        mylcd.lcd_display_string("Progress: %.2f%%" %(percent_cycle), 2)
    endtime = time.time()
    print("\n\n")
    print("\n Output: ")
    print(classify_output)
    print("\nTime elapsed: ")
    print(endtime-starttime)
    
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Time elapsed:", 1)
    mylcd.lcd_display_string("%.2f seconds" %(endtime-starttime), 2)
    time.sleep(3)    
    mylcd.lcd_clear()
    
    listO2V = []
    listMQ8V = []
    listMQ2V = []
    listMQ138V = []
    listCO2V = []
    listMQ136V = []
    listMQ7V = []
    listMQ4V = []
    classname = []
    output = []

    new_listO2V = []
    new_listMQ8V = []
    new_listMQ2V = []
    new_listMQ138V = []
    new_listCO2V = []
    new_listMQ136V = []
    new_listMQ7V = []
    new_listMQ4V = []

    plotO2V = []
    plotMQ8V = []
    plotMQ2V = []
    plotMQ138V = []
    plotCO2V = []
    plotMQ136V = []
    plotMQ7V = []
    plotMQ4V = []

    ResultO2V = 0
    ResultMQ8V = 0
    ResultMQ2V = 0
    ResultMQ138V = 0
    ResultCO2V = 0
    ResultMQ136V = 0
    ResultMQ7V = 0
    ResultMQ4V = 0

    satPoint = []
    satO2V = []
    satMQ8V = []
    satMQ2V = []
    satMQ138V = []
    satCO2V = []
    satMQ136V = []
    satMQ7V = []
    satMQ4V = []


    i = 0
    j = 0
    k = 0
    l = 0
    index = 0
    newindex = 0

    with open('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm6/'+class_to_file, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            listO2V.append(float(row[0]))
            listMQ8V.append(float(row[1]))
            listMQ2V.append(float(row[2]))
            listMQ138V.append(float(row[3]))
            listCO2V.append(float(row[4]))
	    listMQ136V.append(float(row[5]))
            listMQ7V.append(float(row[6]))
            listMQ4V.append(float(row[7]))
            classname.append(str(row[8]))
            output.append(str(row[9]))

    while index < len(listO2V):
        new_listO2V.append(listO2V[index])
        new_listMQ8V.append(listMQ8V[index])
        new_listMQ2V.append(listMQ2V[index])
        new_listMQ138V.append(listMQ138V[index])
        new_listCO2V.append(listCO2V[index])
	new_listMQ136V.append(listMQ136V[index])
        new_listMQ7V.append(listMQ7V[index])
        new_listMQ4V.append(listMQ4V[index])
        index += 50

    while newindex < len(new_listO2V):
        ResultO2V = (new_listO2V[newindex] + new_listO2V[newindex+1])/2
        ResultMQ8V = (new_listMQ8V[newindex] + new_listMQ8V[newindex+1])/2
        ResultMQ2V = (new_listMQ2V[newindex] + new_listMQ2V[newindex+1])/2
        ResultMQ138V = (new_listMQ138V[newindex] + new_listMQ138V[newindex+1])/2
        ResultCO2V = (new_listCO2V[newindex] + new_listCO2V[newindex+1])/2
	ResultMQ136V = (new_listMQ136V[newindex] + new_listMQ136V[newindex+1])/2
        ResultMQ7V = (new_listMQ7V[newindex] + new_listMQ7V[newindex+1])/2
        ResultMQ4V = (new_listMQ4V[newindex] + new_listMQ4V[newindex+1])/2

        plotO2V.append(ResultO2V)
        plotMQ8V.append(ResultMQ8V)
        plotMQ2V.append(ResultMQ2V)
        plotMQ138V.append(ResultMQ138V)
        plotCO2V.append(ResultCO2V)
	plotMQ136V.append(ResultMQ136V)
        plotMQ7V.append(ResultMQ7V)
        plotMQ4V.append(ResultMQ4V)

        newindex += 2

    #contains values of the x and y axes
    x_axis = list(range(len(plotO2V)))
    y_axis = [plotO2V, plotMQ8V, plotMQ2V, plotMQ138V, plotCO2V, plotMQ136V, plotMQ7V, plotMQ4V]

    while j < len(y_axis):
        while i < len(x_axis):
            m = 0
            if i != len(plotO2V)-1:
                b = (x_axis[i+1]-x_axis[i])
                d = (y_axis[j][i+1]-y_axis[j][i])
            if b != 0:
                m = (d)/(b)
                # to check the slopes between two data points
                # print(m)
                if j == 0 and (m < 0.1 and m > -0.1): #obtain only the voltage values with less than 0.03 and greater than -0.03 slopes
                    satO2V.append(y_axis[j][i])
                elif j == 1 and (m < 0.1 and m > -0.1):
                    satMQ8V.append(y_axis[j][i])
                elif j == 2 and (m < 0.1 and m > -0.1):
                    satMQ2V.append(y_axis[j][i])
                elif j == 3 and (m < 0.1 and m > -0.1):
                    satMQ138V.append(y_axis[j][i])
                elif j == 4 and (m < 0.1 and m > -0.1):
                    satCO2V.append(y_axis[j][i])
                elif j == 5 and (m < 0.1 and m > -0.1):
                    satMQ136V.append(y_axis[j][i])
		elif j == 6 and (m < 0.1 and m > -0.1):
                    satMQ7V.append(y_axis[j][i])
                elif j == 7 and (m < 0.1 and m > -0.1):
                    satMQ4V.append(y_axis[j][i])
            i += 1
        i = 0
        j += 1
        
    if len(satO2V) != 0:
        if len(satO2V) < len(x_axis):
            k = l = len(satO2V)
            while k < len(x_axis):
                satO2V.append(satO2V[(l-1)])
                k += 1
            k = None
            l = None

    if len(satMQ8V) != 0:
        if len(satMQ8V) < len(x_axis):
            k = l = len(satMQ8V)
            while k < len(x_axis):
                satMQ8V.append(satMQ8V[(l-1)])
                k += 1
            k = None
            l = None

    if len(satMQ2V) != 0:
        if len(satMQ2V) < len(x_axis):
            k = l = len(satMQ2V)
            while k < len(x_axis):
                satMQ2V.append(satMQ2V[(l-1)])
                k += 1
            k = None
            l = None

    if len(satMQ138V) != 0:
        if len(satMQ138V) < len(x_axis):
            k = l = len(satMQ138V)
            while k < len(x_axis):
                satMQ138V.append(satMQ138V[(l-1)])
                k += 1
            k = None
            l = None

    if len(satCO2V) != 0:
        if len(satCO2V) < len(x_axis):
            k = l = len(satCO2V)
            while k < len(x_axis):
                satCO2V.append(satCO2V[(l-1)])
                k += 1
            k = None
            l = None
			
    if len(satMQ136V) != 0:
        if len(satMQ136V) < len(x_axis):
            k = l = len(satMQ136V)
            while k < len(x_axis):
                satMQ136V.append(satMQ136V[(l-1)])
                k += 1
            k = None
            l = None
			
    if len(satMQ7V) != 0:
        if len(satMQ7V) < len(x_axis):
            k = l = len(satMQ7V)
            while k < len(x_axis):
                satMQ7V.append(satMQ7V[(l-1)])
                k += 1
            k = None
            l = None

    if len(satMQ4V) != 0:
        if len(satMQ4V) < len(x_axis):
            k = l = len(satMQ4V)
            while k < len(x_axis):
                satMQ4V.append(satMQ4V[(l-1)])
                k += 1
            k = None
            l = None
    file_to_saturate = 'saturate'+classify_output+'.txt'
    #write to a file for final processing of data
    sat_Data = open('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm6/'+file_to_saturate, "a+")
    i=0
    class_name = classname[0]
    class_output = output[0]
    while i < len(x_axis):
        sat_Data.write(str(satO2V[i]) +",")
        sat_Data.write(str(satMQ8V[i]) +",")
        sat_Data.write(str(satMQ2V[i]) +",")
        sat_Data.write(str(satMQ138V[i]) +",")
        sat_Data.write(str(satCO2V[i]) +",")
	sat_Data.write(str(satMQ136V[i]) +",")
        sat_Data.write(str(satMQ7V[i]) +",")
        sat_Data.write(str(satMQ4V[i]) +",")
        sat_Data.write(str(class_name) +",")
        sat_Data.write(str(class_output) +"\n")   
        i+=1
    sat_Data.close()
	
    file_to_max = 'max'+classify_output+'.txt'
    maxO2V = max(satO2V)
    maxMQ8V = max(satMQ8V)
    maxMQ2V = max(satMQ2V)
    maxMQ138V = max(satMQ138V)
    maxCO2V = max(satCO2V)
    maxMQ136V = max(satMQ136V)
    maxMQ7V = max(satMQ7V)
    maxMQ4V = max(satMQ4V)
    #write to a file for final processing of data
    max_Data = open('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm6/'+file_to_max, "a+")

    max_Data.write(str(maxO2V) +",")
    max_Data.write(str(maxMQ8V) +",")
    max_Data.write(str(maxMQ2V) +",")
    max_Data.write(str(maxMQ138V) +",")
    max_Data.write(str(maxCO2V) +",")
    max_Data.write(str(maxMQ136V) +",")
    max_Data.write(str(maxMQ7V) +",")
    max_Data.write(str(maxMQ4V) +",")
    max_Data.write(str(class_name) +",")
    max_Data.write(str(class_output) +"\n") 
    max_Data.close()

def plot_data():
    horizontal = []
    plotO2V = []
    plotMQ8V = []
    plotMQ2V = []
    plotMQ138V = []
    plotCO2V = []
    plotMQ136V = []
    plotMQ7V = []
    plotMQ4V = []
    
    file_check_loop = 0
    while file_check_loop == 0:
        classify_output = raw_input("Name of Class for file search: ")
        class_to_file = 'database'+classify_output+'.txt'
        My_file = Path('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm6/'+class_to_file)
        if My_file.is_file():
            print('\nDatabase to plot exist: '+class_to_file+'\n')
            file_check_loop = 1
        else:
            print('\nNo Database to plot named: '+class_to_file+'\n')
            file_check_loop = 0
            
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Plotting...", 1)
    mylcd.lcd_display_string("Class: %s" %(classify_output), 2)
    
    with open('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm6/'+class_to_file,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            plotO2V.append(float(row[0]))
            plotMQ8V.append(float(row[1]))
            plotMQ2V.append(float(row[2]))
            plotMQ138V.append(float(row[3]))
            plotCO2V.append(float(row[4]))
	    plotMQ136V.append(float(row[5]))
            plotMQ7V.append(float(row[6]))
            plotMQ4V.append(float(row[7]))

    plt.plot(plotO2V, 'b', label='O2')
    plt.plot(plotMQ8V, 'g', label='MQ8')
    plt.plot(plotMQ2V, 'r', label='MQ2')
    plt.plot(plotMQ138V, 'c', label='MQ138')
    plt.plot(plotCO2V, 'm', label='CO2')
    plt.plot(plotMQ136V, 'y', label='MQ136')
    plt.plot(plotMQ7V, 'k', label='MQ7')
    plt.plot(plotMQ4V, '0.75', label='MQ4')

    plt.xlabel('Iterations')
    plt.ylabel('Concentrations in V')
    plt.title('Sensor Response')
    plt.legend()
    plt.show()
    mylcd.lcd_clear()
    
# Create an ADS1115 ADC (16-bit) instance.
adc = Adafruit_ADS1x15.ADS1115()
adc2 = Adafruit_ADS1x15.ADS1115(address=0x4a)

mylcd = I2C_LCD_driver.lcd()
mylcd.lcd_clear()
x=0
callibration_check = 0
calcheck=0
###########################################################################
######################### Sensor mq8 on channel 1 #########################
###########################################################################

######################### Hardware Related Macros #########################
RL_VALUE1                     = 10        # define the load resistance on the board, in kilo ohms
RO_CLEAN_AIR_FACTOR1          = 70     # RO_CLEAR_AIR_FACTOR=(Sensor resistance in clean air)/RO,
                                            # which is derived from the chart in datasheet
											
###########################################################################
######################### Sensor mq2 on channel 2 #########################
###########################################################################

######################### Hardware Related Macros #########################
RL_VALUE2                     = 5       # define the load resistance on the board, in kilo ohms
RO_CLEAN_AIR_FACTOR2          = 9.65     		# RO_CLEAR_AIR_FACTOR=(Sensor resistance in clean air)/RO,
                                            # which is derived from the chart in datasheet											
###########################################################################
######################## Sensor mq138 on channel 3 ########################
###########################################################################

######################### Hardware Related Macros #########################
RL_VALUE3                     = 47        # define the load resistance on the board, in kilo ohms
RO_CLEAN_AIR_FACTOR3          = 1     # RO_CLEAR_AIR_FACTOR=(Sensor resistance in clean air)/RO,
                                            # which is derived from the chart in datasheet
                
###########################################################################
######################## Sensor mq135 on channel 4 ########################
###########################################################################

######################### Hardware Related Macros #########################
RL_VALUE4                     = 20      # define the load resistance on the board, in kilo ohms
RO_CLEAN_AIR_FACTOR4          = 3.7     		# RO_CLEAR_AIR_FACTOR=(Sensor resistance in clean air)/RO,
                                            # which is derived from the chart in datasheet
                                            
################################ ADS1115 #2 ###############################
											
###########################################################################
######################## Sensor mq6 on channel 5 ########################
###########################################################################

######################### Hardware Related Macros #########################
RL_VALUE5                     = 20       # define the load resistance on the board, in kilo ohms
RO_CLEAN_AIR_FACTOR5          = 10    		# RO_CLEAR_AIR_FACTOR=(Sensor resistance in clean air)/RO,
                                            # which is derived from the chart in datasheet

###########################################################################
######################## Sensor mq7 on channel 6 ########################
###########################################################################

######################### Hardware Related Macros #########################
RL_VALUE6                     = 10 	# define the load resistance on the board, in kilo ohms
RO_CLEAN_AIR_FACTOR6          = 27     		# RO_CLEAR_AIR_FACTOR=(Sensor resistance in clean air)/RO,
                                            # which is derived from the chart in datasheet
											
###########################################################################
######################## Sensor mq4 on channel 7 ########################
###########################################################################

######################### Hardware Related Macros #########################
RL_VALUE7                     = 20      # define the load resistance on the board, in kilo ohms
RO_CLEAN_AIR_FACTOR7          = 4.5     		# RO_CLEAR_AIR_FACTOR=(Sensor resistance in clean air)/RO,
                                            # which is derived from the chart in datasheet
                                            
######################### Software Related Macros #########################
CALIBARAION_SAMPLE_TIMES     = 50       # define how many samples you are going to take in the calibration phase
CALIBRATION_SAMPLE_INTERVAL  = 500      # define the time interal(in milisecond) between each samples in the
                                        # cablibration phase
READ_SAMPLE_INTERVAL         = 50       # define how many samples you are going to take in normal operation
READ_SAMPLE_TIMES            = 5        # define the time interal(in milisecond) between each samples in 
                                        # normal operation

###########################################################################										
# Or create an ADS1015 ADC (12-bit) instance.
#adc = Adafruit_ADS1x15.ADS1015()

# Note you can change the I2C address from its default (0x48), and/or the I2C
# bus by passing in these optional parameters:
#adc = Adafruit_ADS1x15.ADS1015(address=0x49, busnum=1)

# Choose a gain of 1 for reading voltages from 0 to 4.09V.
# Or pick a different gain to change the range of voltages that are read:
#  - 2/3 = +/-6.144V
#  -   1 = +/-4.096V
#  -   2 = +/-2.048V
#  -   4 = +/-1.024V
#  -   8 = +/-0.512V
#  -  16 = +/-0.256V
# See table 3 in the ADS1015/ADS1115 datasheet for more info on gain.
GAIN = 1

#print('Reading ADS1x15 values, press Ctrl-C to quit...')
# Print nice channel column headers.
#print('| {0:>6} | {1:>6} | {2:>6} | {3:>6} |'.format(*range(4)))
#print('-' * 37)

# Main loop.

#formula is Sensor Voltage = AnalogReading / 4.096V / 32767 if gain is 1
#---------------------------------------------------
channel1 = ((adc.read_adc(1, gain=GAIN))*4.096/32767)     #for mq8 if gain is 1
#slope and coordinates for mq-8 on channel 1:
m1 = -1.44904
x1 = 2.3
y1 = 0.92

#---------------------------------------------------
channel2 = ((adc.read_adc(2, gain=GAIN))*4.096/32767)     #for mq2 if gain is 1
#slope and coordinates for mq-2 on channel 2:
m2 = -0.473054
x2 = 2.3
y2 = 0.32222

#---------------------------------------------------
channel3 = ((adc.read_adc(3, gain=GAIN))*4.096/32767)	  #for mq138 if gain is 1
#slope and coordinates for mq-138 on channel 3:
m3 = -0.425969
x3 = 1.7
y3 = -0.495 

#---------------------------------------------------
channel4 = ((adc2.read_adc(0, gain=GAIN))*4.096/32767)	#for mq135 if gain is 1
#slope and coordinates for mq-135 on channel 2:
m4 = -0.35252
x4 = 1
y4 = 0.3617

channel5 = ((adc2.read_adc(1, gain=GAIN))*4.096/32767)

#---------------------------------------------------
channel6 = ((adc2.read_adc(2, gain=GAIN))*4.096/32767)	  #for mq7 if gain is 1
#slope and coordinates for mq-7 on channel 1:
m6 = -0.738995
x6 = 1.7
y6 = 0.1139

#---------------------------------------------------
channel7 = ((adc2.read_adc(3, gain=GAIN))*4.096/32767)
#for mq4 if gain is 1
#slope and coordinates for mq-6 on channel 2:
m7 = -0.354367
x7 = 2.3
y7 = 0.25527

def countdown_timer():
    
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Purging...", 1)
    x=0

    while x < 1800: #number of seconds
        O2V = ((adc.read_adc(0, gain=GAIN))*4.096/32767)
        MQ8V = ((adc.read_adc(1, gain=GAIN))*4.096/32767)
        MQ2V = ((adc.read_adc(2, gain=GAIN))*4.096/32767)
        MQ138V = ((adc.read_adc(3, gain=GAIN))*4.096/32767)
        CO2V= ((adc2.read_adc(0, gain=GAIN))*4.096/32767)
        MQ136V = ((adc2.read_adc(1, gain=GAIN))*4.096/32767)
        MQ7V = ((adc2.read_adc(2, gain=GAIN))*4.096/32767)
        MQ4V = ((adc2.read_adc(3, gain=GAIN))*4.096/32767)
        
        sys.stdout.write("\r")
        sys.stdout.write("\033[K")
        sys.stdout.write("MQ8: %g V, MQ138: %g V, CO2: %g V, O2: %g V, MQ2: %g V, MQ136: %g V, MQ7: %g V, MQ4: %g V " % (MQ8V, MQ138V, CO2V, O2V, MQ2V, MQ136V, MQ7V, MQ4V))
            
        sys.stdout.flush()
        time.sleep(0.176)
        x=x+1
        percent_cycle=(x/1800)*100
        mylcd.lcd_clear()
        mylcd.lcd_display_string("Purging...", 1)
        mylcd.lcd_display_string("Progress: %.2f%%" %(percent_cycle), 2)

loop_check=0
while loop_check == 0:
    
    mylcd.lcd_clear()
    mylcd.lcd_display_string("Choose an Option", 1)
    mylcd.lcd_display_string("1-R 2-D 3-P 4-E", 2)
    
    print("\n#####################################\n")
    print("(1) Read and Record Data (Again)?\n")
    print("(2) Display Sensor Response Plot?\n")
    print("(3) Purge\n")
    print("(4) Exit\n")
    print("#####################################\n")
    user_input = raw_input("Choose Option: ")
    
    if user_input == '1':
        loop_check=0
        training_read()
    elif user_input == '2':
        plot_data()
        loop_check=0
    elif user_input == '3':
        loop_check=0
        countdown_timer()
    elif user_input == '4':
        loop_check=1
        mylcd.lcd_clear()
        exit()
    else:
        loop_check=0
        print("\n\nWrong Input\n\n")
        mylcd.lcd_clear()
        mylcd.lcd_display_string("Wrong Input", 1)
        time.sleep(2)
        mylcd.lcd_clear()
