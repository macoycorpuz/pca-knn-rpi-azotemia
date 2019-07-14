import time
import Adafruit_ADS1x15

# adc1 = MQ2 to MQ4
# adc2 = MQ6 to MQ135
class SensorHelper:

    sensors = ['MQ2', 'MQ3', 'MQ4', 'MQ6', 'MQ7', 'MQ8', 'MQ135']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    def __init__(self, start_time):
        self.adc1 = Adafruit_ADS1x15.ADS1115(address=0x49)
        self.adc2 = Adafruit_ADS1x15.ADS1115(address=0x48)
        self.start_time = start_time
    
    def print_cli_header(self):
        print('| {0:>7} | {1:>7} | {2:>7} | {3:>7} | {4:>7} | {5:>7} | {6:>7} | {7:>7} |'.format(*sensors))
        print('-' * 74)
        print('\n' * 3)

    def print_cli(self, ctr, values):
        print("\033[F"*5)
        print('| {0:>7} | {1:>7} | {2:>7} | {3:>7} | {4:>7} | {5:>7} | {6:>7} | {7:>7} |\n'.format(*values))
        print('Iteration: {}'.format(ctr))
        print('Elapsed Time: {0:.2f}'.format(time.time()-self.start_time))

    def save_data(self, ):
        for i in range(7):
            adc = adc1.read_adc(i) if i < 4 else adc2.read_adc(i-4)
            volts = adc * (4.096/32767)
            values[i] = round(volts, 5)
            sensor = sensors[i]

if __name__ == '__main__':
    sensor = SensorHelper(time_time())