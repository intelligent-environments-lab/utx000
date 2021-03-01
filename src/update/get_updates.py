import os
import logging

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

class update():

    def __init__():
        self.run([self.update_os,self.upgrade])

    def update_os(self, beacon_no):
        """
        Updates RPi OS
        """
        print("\n\tUpdating Packages:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo apt-get update"')

    def upgrade(self, beacon_no):
        """
        Upgrades RPi libraries
        """
        print("\n\tUpgrading Packages:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo apt-get -y upgrade"')

    def run(self, fxns):
        """
        Runs the functions in fxns
        """
        for i in self.beacon:
            if i < 10:
                i = '0' + str(i)
            #print(f"Beacon {i}:")
            os.system(f'tput setaf 1; echo Beacon {i}; tput sgr 0')
            for f in fxns:
                f(i)

class install_libraries(update):

    def __init__(self):
        self.run()

    def add_pigpio(self, beacon_no):
        """
        Adds the pigpio library for python and python3. The pigpio package is the library used to communicate with the RPi pins.
        """
        print("\n\tAdding PiGPIO:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo apt-get install -y pigpio python-pigpio python3-pigpio"')

    def add_oled(self, beacon_no):
        """
        Adds the libraries needed to control the OLED screens.
        """
        # adding dependent packages
        self.add_pigpio(beacon_no)
        print("\n\tAdding OLED Packages")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo pip3 install oled_text"')

    def add_datascience(self, beacon_no):
        """
        Adds datascience packages numpy and pandas
        """
        print("\n\tAdding NumPy:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo pip install numpy"')
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo pip3 install numpy"')

        print("\n\tAdding Pandas:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo pip install pandas"')
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo pip3 install pandas"')

    def run(self):
        pass

def main():
    pass

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='updates.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()