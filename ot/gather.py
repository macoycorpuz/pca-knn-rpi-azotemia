import time
import Adafruit_ADS1x15
import csv

adc1 = Adafruit_ADS1x15.ADS1115(address=0x48, busnum=1)
adc2 = Adafruit_ADS1x15.ADS1115(address=0x49, busnum=1)

GAIN = 1
sensors = ['MQ135', 'MQ3', 'MQ4', 'MQ2', 'MQ4', 'MQ6', 'MQ7', 'MQ8']

print('Reading ADS1x15 values, press Ctrl-C to quit...')
print("Training Data Name:", end=" ")
name = input()
print('| {0:>7} | {1:>7} | {2:>7} | {3:>7} | {4:>7} | {5:>7} | {6:>7} | {7:>7} |'.format(*sensors))
print('-' * 74)
with open(name + '.csv', mode='w') as csv_file:
    csv_file = csv.writer(csv_file, delimiter=',')
    iteration = 0
    while iteration != 1000:
        iteration += 1
        values = [0]*8
        for i in range(4):
            values[i] = ('{0:.5f}').format(adc1.read_adc(i, gain=GAIN) * (4.096/32767))
        for i in range(4,8):
            values[i] = ('{0:.5f}').format(adc2.read_adc(i-4, gain=GAIN) * (4.096/32767))

        csv_file.writerow(values)
        print('| {0:>7} | {1:>7} | {2:>7} | {3:>7} | {4:>7} | {5:>7} | {6:>7} | {7:>7} |'.format(*values), end="\r\n")
        print(" Iteration: " + str(iteration), end="\r")
        time.sleep(0.2)
