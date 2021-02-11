import os

for folder in os.listdir("/Users/HagenFritz/Projects/utx000/data/raw/utx000/fitbit/"):
	os.system(f"mkdir /Users/HagenFritz/Projects/utx000/data/raw/utx000/fitbit/{folder}/fitbit")
	os.system(f"mv /Users/HagenFritz/Projects/utx000/data/raw/utx000/fitbit/{folder}/* /Users/HagenFritz/Projects/utx000/data/raw/utx000/fitbit/{folder}/fitbit/")