import os
import logging

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

class update():

    def __init__():

    def getUpdates(self, beacon_no):
        """
        Updates/upgrades and then adds any necessary packages
        """

        print("\n\tUpdating Packages:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo apt-get update"')

        print("\n\tUpgrading Packages:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo apt-get -y upgrade"')

        print("\n\tAdding Python3 PiGPIO:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo apt-get install -y pigpio python-pigpio python3-pigpio"')

        print("\n\tAdding OLED Packages")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo pip3 install oled_text"')

        print("\n\tAdding Python2 NumPy:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo pip install numpy"')

def main():


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='updates.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()