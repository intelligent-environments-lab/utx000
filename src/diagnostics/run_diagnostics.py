import os
import logging

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

class Diagnostics():

    def __init__(self, single=False, study="utx000"):
        """
        Checks if we want a single beacon, otherwise runs for all. User can
        download, remove, or run diagnostics.
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

        self.save_dir = f"~/Projects/utx000/data/raw/{study}/beacon/" # directory to save downloaded data to

        # Commands
        # ------------
        # List of options 
        print("Please choose an option from the list below")
        print("\t1. Update from Git")
        print("\t2. Download Data")
        print("\t3. Download and Remove Data")
        print("\t4. Remove Data")
        print("\t5. Check Sensor Connection")
        print("\t6. Run Full Diagnostics")
        op = int(input("\nOption: "))
        # Run diagnostics
        if op == 1:
            fxns = [self.pullFromGit]
        elif op == 2:
            fxns = [self.downloadData]
        elif op == 3:
            fxns = [self.downloadData, self.removeData]
        elif op == 4:
            fxns = [self.removeData]
        elif op == 5:
            fxns = [self.pullFromGit,self.checkSensors]
        elif op == 6:
            fxns = [self.downloadData,self.checkSensors,self.removeData]
        else:
            os.system("clear")
            print("Not a valid choice")
            self.terminated()

        print("What do you wish to do with the devices after?")
        print("\t1. Power Down")
        print("\t2. Reboot")
        print("\t3. Nothing")
        power_op = int(input("\nOption: "))
        if power_op == 1:
            fxns.append(self.shutdown)
        elif power_op == 2:
            fxns.append(self.reboot)
        else:
            pass

        self.run(fxns)

    def pullFromGit(self, beacon_no):
        """
        Pulls in updates from git,
        """

        print("\n\tUpdating from Git:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "cd bevo_iaq/ && git reset --hard && git pull"')
        os.system(f'scp -o ConnectTimeout=1 ~/Projects/utx000/src/diagnostics/fix_number.sh pi@iaq{beacon_no}:/home/pi/fix_number.sh')
        os.system(f'ssh -o ConnectTimeout=1 pi@iaq{beacon_no} "sh /home/pi/fix_number.sh {beacon_no}"')
        
    def downloadData(self, beacon_no):
        """
        Downloads data from specified beacon
        """
        print("\n\tDownloading Data:")
        os.system(f'scp -r -o ConnectTimeout=3 pi@iaq{beacon_no}:/home/pi/DATA/adafruit/ {self.save_dir}B{beacon_no}/')
        os.system(f'scp -r -o ConnectTimeout=3 pi@iaq{beacon_no}:/home/pi/DATA/sensirion/ {self.save_dir}B{beacon_no}/')
        os.system(f'scp -r -o ConnectTimeout=3 pi@iaq{beacon_no}:/home/pi/DATA/ {self.save_dir}B{beacon_no}/')

        print("\n\tRemoving Bad Data:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "sudo rm DATA/*/b00*"')

    def checkSensors(self, beacon_no):
        """
        Checks the operation of the sensors
        """
        print("\n\tChecking Sensor Connection:")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "python3 bevo_iaq/Setup/Code/addresses.py"')

    def removeData(self, beacon_no):
        """
        Removes data from specified beacon
        """
        print("\n\tDeleting Data...")
        #os.system('tput setaf 1; echo REMOVING LOG2/LOG3 DATA; tput sgr 0')
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "cd DATA/adafruit/ && sudo rm *.csv && cd ../sensirion/ && sudo rm *.csv"')
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 "cd DATA/ && sudo rm *.csv"')

    def reboot(self,beacon_no):
        """
        Reboots the system for changes to take effect
        """
        print("\n\tRebooting...")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "sudo reboot"')

    def shutdown(self,beacon_no):
        """
        Reboots the system for changes to take effect
        """
        print("\n\tShutting Down...")
        os.system(f'ssh pi@iaq{beacon_no} -o ConnectTimeout=1 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "sudo shutdown -h now"')

    def run(self,fxns):
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

def main():
    logger = logging.getLogger(__name__)
    os.system("clear")

    print("Run diagnostics for which study:")
    print("\t1. utx000 (default)\n\t2. wcwh_pilot")
    study_op = input("Option: ")
    if study_op.lower() in ["2","wcwh_pilot"]:
        study = "wcwh_pilot"
    else:
        study = "utx000"

    single = input("Individual (y/n)? ")
    if single.lower() in ["y","yes"]:
        d = Diagnostics(True,study=study)
    else:
        d = Diagnostics(False,study=study)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='diagnostics.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()