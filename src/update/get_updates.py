import os
import logging

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

class update():

    def __init__(self, single=False):
        """
        Checks if we want a single beacon, otherwise runs for all. Updates and upgrades RPi on Beacon(s).
        """
        # Getting Beacon Numbers
        # ----------------------
        if single:
            self.beacon = input("Enter Beacon Number (0,50): ") # setting to user input
            try:
                self.beacon = [int(self.beacon)] # single beacon number - 0 will be added in run()
            except ValueError:
                print("You must enter a valid number")
                self.terminated()
        else:
            self.beacon = range(0,51) # setting to list of all beacons

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

    def pull_from_git(self, beacon_no):
        """
        Pulls in updates from git,
        """

        print("\n\tUpdating from Git:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "cd bevo_iaq/ && git reset --hard && git pull"')
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=3  "sudo sh /home/pi/bevo_iaq/fix_number.sh {beacon_no}"')

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

    def terminated(self):
        print("User has terminated program.")
        return False

class install(update):

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

    def add_tailscale(self, beacon_no, auth_key="tskey-3a35f9a673f63fd83e6ef7ba"):
        """adds tailscale vpn to beacons"""
        # strings containing install isntructions 
        transport = "sudo apt-get install apt-transport-https" # install transport
        signing_key = "curl -fsSL https://pkgs.tailscale.com/stable/raspbian/buster.gpg | sudo apt-key add -"
        repo = "curl -fsSL https://pkgs.tailscale.com/stable/raspbian/buster.list | sudo tee /etc/apt/sources.list.d/tailscale.list"
        up = "sudo apt-get update"
        tailscale = "sudo apt-get install tailscale"
        auth = f"sudo tailscale up --authkey {auth_key}"
        show_key = "ip addr show tailscale"

        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=3 "{transport} && {signing_key} && {repo} && {up} && {tailscale} && {auth} && {show_key}"')

    def add_reboot(self, beacon_no):
        """runs and adds reboot to crontab"""
        print("\n\tAdding reboot functionality to crontab:")
        print()
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=3 "sudo sh /home/pi/bevo_iaq/reboot_install.sh"')

def main():
    up = input("Update and upgrade devices (y/n): ")
    if up.lower() in ["y","yes","1"]:
        single = input("Single beacon or all:\n\t1. single\n\t2. all\nChoice: ")
        if single.lower() in ["1","single","y","yes"]:
            updater = update(single=True)
        else:
            updater = update(single=False)
        
        updater.run([updater.update_os,updater.upgrade])

    ins = input("Install new packages (y/n): ")
    if ins.lower() in ["y","yes","1"]:
        single = input("Single beacon or all:\n\t1. single\n\t2. all\nChoice: ")
        if single.lower() in ["1","single","y","yes"]:
            installer = install(single=True)
        else:
            installer = install(single=False)
        
        # edit with packages to install
        installer.run([installer.pull_from_git,installer.add_reboot])

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='updates.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()