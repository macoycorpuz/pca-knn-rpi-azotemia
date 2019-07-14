##########################################
##########################################
##for training and testing concentatons
##ANNPCAGUI to comm 5
##remove drivers (lcd , dht, ads)
##removed unnecessary sensors (mq138, mq2, o2, mq4, mq135)
##10/11/2018
## affected values : retain data $9,$10 to $9,$10
## cut -f9 to -4in paste1
## reverted back to 8 sensors 11/11/2018
##########################################
##########################################

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk ,GObject

from math import exp
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import time
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
##import Adafruit_ADS1x15
##import Adafruit_DHT
import csv
import sys
##import I2C_LCD_driver
import os
import os.path
import cPickle as pickle
import ast
from pathlib import Path
from sklearn.decomposition import PCA

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def load_csv_2(filename,columns,pc):
        if pc == 1:
            column1=columns[0]
            retain_script='cut -f'+str(column1)+" -d',' "+filename+' > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput2.txt'
            os.system(retain_script)
        elif pc == 2:
            column1=columns[0]
            column2=columns[1]
            retain_script='cut -f'+str(column1)+','+str(column2)+" -d',' "+filename+' > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput2.txt'
            os.system(retain_script)
        elif pc == 3:
            column1=columns[0]
            column2=columns[1]
            column3=columns[2]
            retain_script='cut -f'+str(column1)+','+str(column2)+','+str(column3)+" -d',' "+filename+' > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput2.txt'
            os.system(retain_script)
        elif pc == 4:
            column1=columns[0]
            column2=columns[1]
            column3=columns[2]
            column4=columns[3]
            retain_script='cut -f'+str(column1)+','+str(column2)+','+str(column3)+','+str(column4)+" -d',' "+filename+' > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput2.txt'
            os.system(retain_script)
        elif pc == 5:
            column1=columns[0]
            column2=columns[1]
            column3=columns[2]
            column4=columns[3]
            column5=columns[4]
            retain_script='cut -f'+str(column1)+','+str(column2)+','+str(column3)+','+str(column4)+','+str(column5)+" -d',' "+filename+' > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput2.txt'
            os.system(retain_script)
        elif pc == 6:
            column1=columns[0]
            column2=columns[1]
            column3=columns[2]
            column4=columns[3]
            column5=columns[4]
            column6=columns[5]
            retain_script='cut -f'+str(column1)+','+str(column2)+','+str(column3)+','+str(column4)+','+str(column5)+','+str(column6)+" -d',' "+filename+' > /home/pi/EnoseProto/Enose/examples/Sensor/Dataabase/Comm5/traininginput2.txt'
            os.system(retain_script)
        elif pc == 7:
            column1=columns[0]
            column2=columns[1]
            column3=columns[2]
            column4=columns[3]
            column5=columns[4]
            column6=columns[5]
            column7=columns[6]
            retain_script='cut -f'+str(column1)+','+str(column2)+','+str(column3)+','+str(column4)+','+str(column5)+','+str(column6)+','+str(column7)+" -d',' "+filename+' > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput2.txt'
            os.system(retain_script)
			
	elif pc == 8:
            column1=columns[0]
            column2=columns[1]
            column3=columns[2]
            column4=columns[3]
            column5=columns[4]
            column6=columns[5]
            column7=columns[6]
	    column8=columns[7]
            retain_script='cut -f'+str(column1)+','+str(column2)+','+str(column3)+','+str(column4)+','+str(column5)+','+str(column6)+','+str(column7)+','+str(column8)+" -d',' "+filename+' > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput2.txt'
            os.system(retain_script)

	dataset = list()
	with open('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput2.txt', 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
    
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
			
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
        print('\nTraining the Network...\n')
        
        i_epoch = 0
        progress = 0
	for epoch in range(n_epoch):
                i_epoch = i_epoch+1
                progress=(i_epoch/float(n_epoch))*100
                
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
##		mylcd.lcd_clear()
##                mylcd.lcd_display_string('Progress: %.2f%%' %(progress), 1)
##                mylcd.lcd_display_string('Error: %.2f%%' %(sum_error), 2)
##        mylcd.lcd_clear()
##        mylcd.lcd_display_string("Finished Training", 1)
##        mylcd.lcd_display_string("Error: %.2f%%" %(sum_error), 2)
        return sum_error
# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
    
class Start(Gtk.Window):
    
    def __init__(self):
        Gtk.Window.__init__(self, title="Electronic Nose Training and Testing")
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_default_size(500,100)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)
        
        label = Gtk.Label("Would You like to start the Enose training with the current Database?")
        vbox.pack_start(label, True, True, 0)
        
        button = Gtk.Button.new_with_label("Yes")
        button.connect("clicked", self.on_yes_clicked)
        vbox.pack_start(button, True, True, 1)
        
        button = Gtk.Button.new_with_label("No")
        button.connect("clicked", self.on_no_clicked)
        vbox.pack_start(button, True, True, 1)
        
    def on_yes_clicked(self,button):
        Start.destroy(self)
        
    def on_no_clicked(self,button):
        exit()
        
class ProgressPCA1(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Electronic Nose Training and Testing")
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_default_size(500,100)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)
    
        self.progressbar = Gtk.ProgressBar()
        vbox.pack_start(self.progressbar, True, True, 0)
        self.progressbar.set_text("Running PCA (Loading Dataset)...")
        self.progressbar.set_show_text("Running PCA (Loading Dataset)...")
        self.timeout_id = GObject.timeout_add(5, self.on_timeout, None)
        self.activity_mode = False
        ##mylcd.lcd_clear()
        ##mylcd.lcd_display_string("Running PCA", 1)
        ##mylcd.lcd_display_string("Loading Dataset", 2)
    def destroy_window(self):
        ProgressPCA1.destroy(self)
        
    def on_timeout(self, user_data):
        """
        Update value on the progress bar
        """
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            if self.progressbar.get_fraction() == 1:
                self.destroy_window()
            else:
                new_value = self.progressbar.get_fraction() + 0.01
                self.progressbar.set_fraction(new_value)
        return True

class ProgressPCA2(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="ProgressBar Demo2")
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_default_size(500,100)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(vbox)
    
        self.progressbar = Gtk.ProgressBar()
        vbox.pack_start(self.progressbar, True, True, 0)
        self.progressbar.set_text("Running PCA (Transforming Data)...")
        self.progressbar.set_show_text("Running PCA (Transforming Data)...")
        self.timeout_id = GObject.timeout_add(5, self.on_timeout, None)
        self.activity_mode = False
        ##mylcd.lcd_clear()
        ##mylcd.lcd_display_string("Running PCA", 1)
        ##mylcd.lcd_display_string("Transformed data", 2)
    def destroy_window(self):
        ProgressPCA1.destroy(self)
        
    def on_timeout(self, user_data):
        """
        Update value on the progress bar
        """
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            if self.progressbar.get_fraction() == 1:
                self.destroy_window()
            else:
                new_value = self.progressbar.get_fraction() + 0.01
                self.progressbar.set_fraction(new_value)
        return True
    
class DisplaySensor(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Electronic Nose Training and Testing")
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_default_size(500,100)
        hbox = Gtk.Box(spacing=10)
        hbox.set_homogeneous(False)
        vbox_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_left.set_homogeneous(False)
        vbox_right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_right.set_homogeneous(False)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        
        hbox.pack_start(vbox_left, True, True, 0)
        hbox.pack_start(vbox_right, True, True, 0)
        vbox.pack_start(hbox,True,True,0)
        
        label = Gtk.Label()
        label.set_markup("<b>Effective Sensors: </b>")
        vbox_left.pack_start(label, True, True, 0)
        
        label = Gtk.Label()
        label.set_markup("<b>Retained Informations: </b>")
        vbox_left.pack_start(label, True, True, 0)
        
        label = Gtk.Label()
        label.set_text("%s , %s" %(title[columnmq[0]-1],title[columnmq[1]-1]))
        vbox_right.pack_start(label, True, True, 0)
        
        label = Gtk.Label()
        label.set_text("%.2f%%" %cumval)
        vbox_right.pack_start(label, True, True, 0)
##        self.add(hbox)
        
        button = Gtk.Button.new_with_label("Continue")
        button.connect("clicked", self.on_continue_clicked)
        vbox.pack_start(button, True, True, 0)
        
        self.add(vbox)
        
        ##
        ##mylcd.lcd_clear()
        ##mylcd.lcd_display_string("Effective Sensor", 1)
        ##mylcd.lcd_display_string("%s , %s" %(title[columnmq[0]],title[columnmq[1]]), 2)
        ##time.sleep(2)
        ##
        ##mylcd.lcd_clear()
        ##mylcd.lcd_display_string("Retained Info:", 1)
        ##mylcd.lcd_display_string("%.2f%%" %cumval, 2)
        ##time.sleep(2)
    def on_continue_clicked(self,button):
        Start.destroy(self)
    
class TrainingEpoch(Gtk.Window):
    
    def __init__(self):
        Gtk.Window.__init__(self, title="Electronic Nose Training and Testing")
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_default_size(500,100)
        hbox = Gtk.Box(spacing=10)
        hbox.set_homogeneous(False)
        vbox_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_left.set_homogeneous(False)
        vbox_right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_right.set_homogeneous(False)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        
        hbox.pack_start(vbox_left, True, True, 0)
        hbox.pack_start(vbox_right, True, True, 0)
        vbox.pack_start(hbox,True,True,0)
        
        label = Gtk.Label()
        label.set_markup("<b>Training Epoch: </b>")
        vbox_left.pack_start(label, True, True, 0)
        
        adjustment = Gtk.Adjustment(0, 0, 10000, 1, 10, 0)
        self.spinbutton = Gtk.SpinButton()
        self.spinbutton.set_adjustment(adjustment)
        self.spinbutton.set_numeric(self.spinbutton)
        vbox_right.pack_start(self.spinbutton, True, True, 0)
        
        button = Gtk.Button.new_with_label("Train")
        button.connect("clicked", self.on_yes_clicked)
        vbox.pack_start(button, True, True, 1)
        
        self.add(vbox)
        
    def on_yes_clicked(self,button):
##        self.timeout_id = GObject.timeout_add(5, self.on_timeout, None)
        ##mylcd.lcd_clear()
        ##mylcd.lcd_display_string("Running ANN", 1)
        ##mylcd.lcd_display_string("Training Dataset", 2)
        TrainingEpoch.destroy(self)
    
class TrainingProgress(Gtk.Window):
    
    def __init__(self):
        Gtk.Window.__init__(self, title="Electronic Nose Training and Testing")
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_default_size(500,100)
        
        hbox = Gtk.Box(spacing=10)
        hbox.set_homogeneous(False)
        vbox_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_left.set_homogeneous(False)
        vbox_right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_right.set_homogeneous(False)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        
        hbox.pack_start(vbox_left, True, True, 0)
        hbox.pack_start(vbox_right, True, True, 0)
        vbox.pack_start(hbox,True,True,0)
        
        label = Gtk.Label()
        label.set_markup("<b>Finished Training</b>")
        vbox.pack_start(label, True, True, 0)
        
        label1 = Gtk.Label()
        label1.set_markup("<b>Error: %.2f%%</b>" %(sum_error))
        vbox.pack_start(label1, True, True, 0)
        
        button = Gtk.Button.new_with_label("Train")
        button.connect("clicked", self.on_yes_clicked)
        vbox.pack_start(button, True, True, 1)
        
        self.add(vbox)
        
    def on_yes_clicked(self,button):
        TrainingProgress.destroy(self)
        
class Testing(Gtk.Window):
    
    def __init__(self):
        Gtk.Window.__init__(self, title="Electronic Nose Training and Testing")
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_default_size(500,100)
        
        hbox = Gtk.Box(spacing=10)
        hbox.set_homogeneous(False)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_left.set_homogeneous(False)
        vbox_right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_right.set_homogeneous(False)
        
        vbox.pack_start(hbox,True,True,0)
        hbox.pack_start(vbox_left, True, True, 0)
        hbox.pack_start(vbox_right, True, True, 0)
        
        label = Gtk.Label()
        label.set_markup("Type the Class to Test:")
        vbox.pack_start(label, True, True, 0)
        
        self.entry = Gtk.Entry()
        vbox.pack_start(self.entry, True, True, 0)
        
        button = Gtk.Button.new_with_label("Test")
        button.connect("clicked", self.on_yes_clicked)
        vbox_left.pack_start(button, True, True, 1)
        
        button = Gtk.Button.new_with_label("Exit")
        button.connect("clicked", self.on_exit_clicked)
        vbox_right.pack_start(button, True, True, 1)
        
        self.add(vbox)
        
    def on_yes_clicked(self,button):
        global value_entered
        value_entered= self.entry.get_text()
        self.entry.set_editable(False)
        print 'Class: %s' % value_entered
        Testing.destroy(self)
        
    def on_exit_clicked(self,button):
        exit()
        
class Output(Gtk.Window):
    
    def __init__(self):
        Gtk.Window.__init__(self, title="Electronic Nose Training and Testing")
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_default_size(500,100)
        hbox = Gtk.Box(spacing=10)
        hbox.set_homogeneous(False)
        vbox_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_left.set_homogeneous(False)
        vbox_right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_right.set_homogeneous(False)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        
        hbox.pack_start(vbox_left, True, True, 0)
        hbox.pack_start(vbox_right, True, True, 0)
        vbox.pack_start(hbox,True,True,0)
        
        label = Gtk.Label()
        label.set_markup("Class: %s" %(classify_output))
        vbox.pack_start(label, True, True, 0)
        
        label = Gtk.Label()
        label.set_markup("Output: <b>%s</b>" %(Poutput))
        vbox.pack_start(label, True, True, 0)
        
        button = Gtk.Button.new_with_label("Test")
        button.connect("clicked", self.on_yes_clicked)
        vbox.pack_start(button, True, True, 1)
        
        self.add(vbox)
        
    def on_yes_clicked(self,button):
        Output.destroy(self)
    
class OutputError(Gtk.Window):
    
    def __init__(self):
        Gtk.Window.__init__(self, title="Electronic Nose Training and Testing")
        self.set_border_width(10)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_default_size(500,100)
        hbox = Gtk.Box(spacing=10)
        hbox.set_homogeneous(False)
        vbox_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_left.set_homogeneous(False)
        vbox_right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_right.set_homogeneous(False)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        
        hbox.pack_start(vbox_left, True, True, 0)
        hbox.pack_start(vbox_right, True, True, 0)
        vbox.pack_start(hbox,True,True,0)
        
        label = Gtk.Label()
        label.set_markup("Class Not Available:<b> %s</b>" %(classify_output))
        vbox.pack_start(label, True, True, 0)
        
        button = Gtk.Button.new_with_label("Test Another Class")
        button.connect("clicked", self.on_yes_clicked)
        vbox.pack_start(button, True, True, 1)
        
        self.add(vbox)
        
    def on_yes_clicked(self,button):
        Output.destroy(self)

##mylcd = I2C_LCD_driver.lcd()
##mylcd.lcd_clear()
start = Start()
start.connect("destroy", Gtk.main_quit)
start.show_all()
Gtk.main()

PCA1 = ProgressPCA1()
PCA1.connect("destroy", Gtk.main_quit)
PCA1.show_all()
Gtk.main()
# Test training backprop algorithm
seed(1)

time.sleep(2)
np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:f}'.format})
np.set_printoptions(threshold=np.nan)
df = pd.read_csv('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt', 
    header=None, 
    sep=',')
df.columns = ['O2', 'MQ8', 'MQ2', 'MQ138', 'MQ135', 'MQ136', 'MQ7', 'MQ4', 'class', 'output']
#df.columns = ['MQ8', 'MQ136', 'MQ7', 'class', 'output']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

print(df)

X = df.ix[:,0:8].values
data = df.ix[:,0:8].values
y = df.ix[:,8].values

type = []
#for three sensors :
#os.system("cut -f5 -d',' /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt|sort|uniq > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/class.txt")
#for 8 sensors :
os.system("cut -f10 -d',' /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt|sort|uniq > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/class.txt")
with open('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/class.txt','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in plots:
        type.append(str(row[0]))

X_std = StandardScaler().fit_transform(X)

#covariance matrix
print("For Covariance matrix: \n")
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#correlation matrix
print("For Correlation matrix: \n")
cor_mat1 = np.corrcoef(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#Eigenvalues from highest to lowest in order choose the top k eigenvectors

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

#=============================================================================
#rank the eigenvalues from highest to lowest in order choose the top k eigenvectors

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
#=============================================================================
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

print("\nInformation of Explained Variances of Individual Principal Components (over 100%)\n")
print(var_exp)
print("\nCumulative Explained Variances of PC's\n")
print(cum_var_exp)

matrix_w = np.hstack((eig_pairs[0][1].reshape(8,1), 
                      eig_pairs[1][1].reshape(8,1)))

PCA2 = ProgressPCA2()
PCA2.connect("destroy", Gtk.main_quit)
PCA2.show_all()
Gtk.main()

##print('\nMatrix W:\n%s' %matrix_w)

Y = X_std.dot(matrix_w)

#print('\n\nCoordinates of PCA:\n%s' %Y)

z_scaler = StandardScaler()
z_data = z_scaler.fit_transform(data)
pca_trafo = PCA().fit(z_data);

fig, ax1 = plt.subplots(figsize = (10,6.5))
ax1.semilogy(pca_trafo.explained_variance_ratio_, '--o', label = 'explained variance ratio');
color =  ax1.lines[0].get_color()
ax1.set_xlabel('principal component', fontsize = 20);
plt.legend(loc=(0.01, 0.075) ,fontsize = 18);

ax2 = ax1.twinx()
ax2.semilogy(pca_trafo.explained_variance_ratio_.cumsum(), '--go', label = 'cumulative explained variance ratio');
for tl in ax2.get_yticklabels():
    tl.set_color('g')

ax1.tick_params(axis='both', which='major', labelsize=18);
ax1.tick_params(axis='both', which='minor', labelsize=12);
ax2.tick_params(axis='both', which='major', labelsize=18);
ax2.tick_params(axis='both', which='minor', labelsize=12);
plt.xlim([0,8]);
plt.legend(loc=(0.01, 0),fontsize = 18);

#mean and variance curve

iter=0
pc=2
cumval = 0
while cumval < 90:
    cumval = cum_var_exp[iter]
    iter=iter+1
##    pc=pc+1
print('\nNumber of PCs: %s' %pc)

n_comp = pc
pca_trafo = PCA(n_components=n_comp)

z_scaler = StandardScaler()
z_data = z_scaler.fit_transform(data)

pca_data = pca_trafo.fit_transform(z_data)
pca_inv_data = pca_trafo.inverse_transform(np.eye(n_comp))

fig = plt.figure(figsize=(10, 6.5))
plt.plot(pca_inv_data.mean(axis=0), '--o', label = 'mean')
plt.plot(np.square(pca_inv_data.std(axis=0)), '--o', label = 'variance')
plt.legend(loc='lower right')
plt.ylabel('feature contribution', fontsize=20);
plt.xlabel('feature index (O2,MQ8,MQ2,MQ138,MQ135,MQ136,MQ7,MQ4)', fontsize=20);
plt.tick_params(axis='both', which='major', labelsize=18);
plt.tick_params(axis='both', which='minor', labelsize=12);
plt.xlim([0, 8])
plt.legend(loc='lower left', fontsize=18)
plt.show()

pca_mean = pca_inv_data.mean(axis=0)[::-1]
pca_variance = np.square(pca_inv_data.std(axis=0))[::-1]
pca_two_mean_max=pca_mean.argsort()[-(n_comp):][::-1]
pca_two_variance_max=pca_variance.argsort()[-(n_comp):][::-1]

##pca_two_mean_max=np.sort(pca_two_mean_max)
##
##pca_two_variance_max=np.sort(pca_two_variance_max)

print('\nMean \n%s' %pca_mean)
print('\nVariance \n%s' %pca_variance)
print('\nMax Feature Mean \n%s' %pca_two_mean_max)
print('\nMax Feature Variance \n%s' %pca_two_variance_max)
columnmq = []
#for 8 sensors:
title = np.array(['O2', 'MQ8', 'MQ2', 'MQ138', 'MQ135', 'MQ136', 'MQ7', 'MQ4'])
#for 3 sensors:
#title = np.array(['MQ8', 'MQ136', 'MQ7'])

for i in pca_two_variance_max:
    print('Sensors: %s' %title.item(i))
    increment = i+1
    columnmq.append(increment)
    
Disp = DisplaySensor()
Disp.connect("destroy", Gtk.main_quit)
Disp.show_all()
Gtk.main()

if pc == 1:
    column1=columnmq[0]
    retain_script = "awk -F, '{print $"+str(column1)+",$9,$10}' OFS=, /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt"
    os.system(retain_script)
elif pc == 2:
    column1=columnmq[0]
    column2=columnmq[1]
    retain_script = "awk -F, '{print $"+str(column1)+',$'+str(column2)+",$9,$10}' OFS=, /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt"
    os.system(retain_script)
elif pc == 3:
    column1=columnmq[0]
    column2=columnmq[1]
    column3=columnmq[2]
    retain_script = "awk -F, '{print $"+str(column1)+',$'+str(column2)+',$'+str(column3)+",$9,$10}' OFS=, /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt"
    os.system(retain_script)
elif pc == 4:
    column1=columnmq[0]
    column2=columnmq[1]
    column3=columnmq[2]
    column4=columnmq[3]
    retain_script = "awk -F, '{print $"+str(column1)+',$'+str(column2)+',$'+str(column3)+',$'+str(column4)+",$9,$10}' OFS=, /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt"
    os.system(retain_script)
elif pc == 5:
    column1=columnmq[0]
    column2=columnmq[1]
    column3=columnmq[2]
    column4=columnmq[3]
    column5=columnmq[4]
    retain_script = "awk -F, '{print $"+str(column1)+',$'+str(column2)+',$'+str(column3)+',$'+str(column4)+',$'+str(column5)+",$9,$10}' OFS=, /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt"
    os.system(retain_script)
elif pc == 6:
    column1=columnmq[0]
    column2=columnmq[1]
    column3=columnmq[2]
    column4=columnmq[3]
    column5=columnmq[4]
    column6=columnmq[5]
    retain_script = "awk -F, '{print $"+str(column1)+',$'+str(column2)+',$'+str(column3)+',$'+str(column4)+',$'+str(column5)+',$'+str(column6)+",$9,$10}' OFS=, /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt"
    os.system(retain_script)
elif pc == 7:
    column1=columnmq[0]
    column2=columnmq[1]
    column3=columnmq[2]
    column4=columnmq[3]
    column5=columnmq[4]
    column6=columnmq[5]
    column7=columnmq[6]
    retain_script = "awk -F, '{print $"+str(column1)+',$'+str(column2)+',$'+str(column3)+',$'+str(column4)+',$'+str(column5)+',$'+str(column6)+',$'+str(column7)+",$9,$10}' OFS=, /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt"
    os.system(retain_script)
	
elif pc == 8:
    column1=columnmq[0]
    column2=columnmq[1]
    column3=columnmq[2]
    column4=columnmq[3]
    column5=columnmq[4]
    column6=columnmq[5]
    column7=columnmq[6]
    column8=columnmq[7]
    retain_script = "awk -F, '{print $"+str(column1)+',$'+str(column2)+',$'+str(column3)+',$'+str(column4)+',$'+str(column5)+',$'+str(column6)+',$'+str(column7)+',$'+str(column8)+",$9,$10}' OFS=, /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/maxAcetone.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt"
    os.system(retain_script)

## record the class into a file named paste 1.txt
os.system("cut -f4 -d',' /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/paste1.txt")

## change "cut -f1-2" if more than 2 pc or 1 pc
os.system("cut -f1-2 -d',' /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/retaindata.txt|sed -e 's/$/,/'> /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/paste2.txt")

## combine the first and second paste file
os.system("paste /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/paste2.txt /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/paste1.txt > /home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput.txt")

#filename = '/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm/database2.txt'
filename = '/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/traininginput.txt'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)
##minmax = dataset_minmax(dataset)
##normalize_dataset(dataset, minmax)

##mylcd.lcd_clear()
##mylcd.lcd_display_string("Running ANN", 1)
##mylcd.lcd_display_string("Training Dataset", 2)
Tpoc = TrainingEpoch()
Tpoc.connect("destroy", Gtk.main_quit)
Tpoc.show_all()
Gtk.main()

n_folds = 5
l_rate = 0.3
n_epoch = Tpoc.spinbutton.get_value_as_int()
n_hidden = 5

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, n_hidden, n_outputs)
sum_error=train_network(network, dataset, l_rate, n_epoch, n_outputs)

Trpo = TrainingProgress()
Trpo.connect("destroy", Gtk.main_quit)
Trpo.show_all()
Gtk.main()
        
for layer in network:
	print(layer)

file_check_loop = 0
while file_check_loop == 0:
##    mylcd.lcd_clear()
##    mylcd.lcd_display_string("Select Class", 1)
    Comb = Testing()
    Comb.connect("destroy", Gtk.main_quit)
    Comb.show_all()
    Gtk.main()
    
    classify_output = value_entered
    print(classify_output)
##    mylcd.lcd_display_string("Class: %s" %(classify_output), 2)
##    time.sleep(1)
    class_to_file = 'saturate'+classify_output+'.txt'
    My_file = Path('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/'+class_to_file)
    if My_file.is_file():
        print('\nDatabase to Test exist: '+class_to_file+'\n')
        file_check_loop = 0
        dataset1 = load_csv_2('/home/pi/EnoseProto/Enose/examples/Sensor/Database/Comm5/'+class_to_file, columnmq,pc)
        for i in range(len(dataset1[0])):
            str_column_to_float(dataset1, i)
	for row in dataset1:
            prediction = predict(network, row)
            if prediction == 0:
                Poutput = 'Precursor'
                print(Poutput)
##                mylcd.lcd_clear()
##                mylcd.lcd_display_string("Class: %s" %(classify_output), 1)
##                mylcd.lcd_display_string("Precursor", 2)
##                time.sleep(10)
            elif prediction == 1:
                Poutput = 'Non-Precursor'
                print(Poutput)
##                mylcd.lcd_clear()
##                mylcd.lcd_display_string("Class: %s" %(classify_output), 1)
##                mylcd.lcd_display_string("Non-precursor", 2)
##                time.sleep(10)
        Outp = Output()
        Outp.connect("destroy", Gtk.main_quit)
        Outp.show_all()
        Gtk.main()
    else:
        print('\nNo Database to Test named: '+class_to_file+'\n')
        Outp = OutputError()
        Outp.connect("destroy", Gtk.main_quit)
        Outp.show_all()
        Gtk.main()
        file_check_loop = 0
        
##Comb = Testing()
##Comb.connect("destroy", Gtk.main_quit)
##Comb.show_all()
##Gtk.main()

widget = Gtk.Box()
print(dir(widget.props))

