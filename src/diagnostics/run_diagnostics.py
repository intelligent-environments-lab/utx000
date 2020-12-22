import os
import logging

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

class Diagnostics():
	def __init__(self, single=False, save_dir="../../data/raw/bpeace2/beacon/"):
		if single:
			self.beacon = input("Enter Beacon Number (0,50): ") # setting to user input
			try:
				if int(self.beacon) < 10:
					self.beacon = '0' + str(self.beacon)
			except ValueError:
				print("You must enter a valid number")
				self.terminate()
		else:
			self.beacon = "All" # setting to dummy

		self.single = single
		self.save_dir = save_dir

		op1 = input(f"Run Diagnostics on Beacon {self.beacon} (y/n)? ")
		if op1.lower() in ["y","yes"]:
			print(f"WARNING: Running diagnostics will remove all data from the selected Beacons and save the data to this directory:\n\t{self.save_dir}")
			proceed = input("Do you still wish to proceed (y/n)? ")
			if proceed.lower() in ["y","yes"]:
				self.run()
			else:
				self.terminated()
		else:
			op2 = input(f"Download data from Beacon {self.beacon} only (y/n)? ")
			if op2.lower() in ["y","yes"]:
				self.downloadData(save_dir=self.save_dir)
			else:
				op3 = input(f"Download data from Beacon {self.beacon} and DELETE after receiving (y/n)? ")
				if op3.lower() in ["y","yes"]:
					print(f"WARNING: Running this option will remove all data from the selected Beacons and save the data to this directory:\n\t{self.save_dir}")
					proceed = input("Do you still wish to proceed (y/n)? ")
					if proceed.lower() in ["y","yes"]:
						self.downloadData(remove=True,save_dir=self.save_dir)
					else:
						self.terminated()
				else:
					self.terminated()


	def downloadData(self, remove=False, save_dir="../../data/raw/bpeace2/beacon/"):
		"""
		Downloads data from Beacons

		Inputs:
		- remove: boolean specifying whether or not to delete data afterwards
		- save_dir: string specifying where to save the data

		Returns void
		"""
		os.system("clear")
		print("Downloading Data...")
		if self.single:
			os.system(f'scp -r -o ConnectTimeout=1 pi@iaq{self.beacon}:/home/pi/DATA/adafruit/ {save_dir}B{self.beacon}/')
			os.system(f'scp -r -o ConnectTimeout=1 pi@iaq{self.beacon}:/home/pi/DATA/sensirion/ {save_dir}B{self.beacon}/')
		else:
			for i in range(0,51):
				if i < 10:
					i = '0' + str(i)

				print(f"Beacon {i}:")
				os.system(f'scp -r -o ConnectTimeout=1 pi@iaq{i}:/home/pi/DATA/adafruit/ {save_dir}B{i}/')
				os.system(f'scp -r -o ConnectTimeout=1 pi@iaq{i}:/home/pi/DATA/sensirion/ {save_dir}B{i}/')

		if remove:
			self.removeData()

	def removeData(self):
		"""
		Removes data from Beacons

		Returns void
		"""
		os.system("clear")
		print("Removing Data...")
		if self.single:
			os.system('tput setaf 1; echo REMOVING LOG2 DATA; tput sgr 0')
			os.system(f'ssh pi@iaq{self.beacon} -o ConnectTimeout=10 "cd DATA/adafruit/ && sudo rm *.csv"')
			os.system('tput setaf 1; echo REMOVING LOG3 DATA; tput sgr 0')
			os.system(f'ssh pi@iaq{self.beacon} -o ConnectTimeout=10 "cd DATA/sensirion/ && sudo rm *.csv"')
		else:
			for i in range(0,51): # looping through all the beacons
				if i < 10:
					i = '0' + str(i)

				print(f"Beacon {i}:")
				print('\tDeleting log2 data')
				os.system(f'ssh pi@iaq{i} -o ConnectTimeout=10 "cd DATA/adafruit/ && sudo rm *.csv"')
				print('\tDeleting log3 data')
				os.system(f'ssh pi@iaq{i} -o ConnectTimeout=10 "cd DATA/sensirion/ && sudo rm *.csv"')

	def checkSensors(self):
		"""
		Checks the operation of the sensors
		"""

	def run(self):
		"""
		Runs diagnostics
		"""

	def terminated(self):
		print("User has terminated program.")
		os.system("clear")
		return False

def main():
	logger = logging.getLogger(__name__)
	os.system("clear")

	single = input("Individual Download (y/n)? ")
	if single.lower() in ["y","yes"]:
		d = Diagnostics(True)
	else:
		d = Diagnostics(False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='diagnostics.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()