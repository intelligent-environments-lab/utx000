import os

import pandas as pd
import numpy as np

from datetime import datetime

class Internal_Temperature_Check():

    def __init__(self, study="wcwh_pilot", suffix="ux_s20", data_dir="../data/",beacons=np.arange(0,51,1), start_time=datetime(2020,1,1),end_time=datetime(2022,1,1)):
        """
        Initializes the class
        """
        self.study = study
        self.suffix = suffix
        self.beacons = beacons
        self.data_dir = data_dir

        self.start_time = start_time
        self.end_time = end_time

        self.set_correction_model()

    def set_data(self, verbose=False, **kwargs):
        """
        sets beacon data for analysis
        """
        data = pd.DataFrame()
        for beacon in self.beacons:
            number = f'{beacon:02}'
            data_by_beacon = pd.DataFrame()
            if verbose:
                print("Beacon", beacon)
            try:
                for file in os.listdir(f"{self.data_dir}raw/{self.study}/beacon/B{number}/DATA/"):
                    if file[-1] == "v":
                        y = int(file.split("_")[1].split("-")[0])
                        m = int(file.split("_")[1].split("-")[1])
                        d = int(file.split("_")[1].split("-")[2].split(".")[0])
                        date = datetime(y,m,d)
                        if date.date() >= self.start_time.date() and date.date() <= self.end_time.date():
                            try:
                                temp = pd.read_csv(f"{self.data_dir}raw/{self.study}/beacon/B{number}/DATA/{file}")
                                if len(temp) > 0:
                                    data_by_beacon = data_by_beacon.append(temp)
                            except Exception as e:
                                print("Error with file", file+":", e)
                if len(data_by_beacon) > 0:
                    data_by_beacon["Timestamp"] = pd.to_datetime(data_by_beacon["Timestamp"])
                    data_by_beacon = data_by_beacon.dropna(subset=["Timestamp"]).set_index("Timestamp").sort_index()[self.start_time:self.end_time].resample("1T").mean()
                    data_by_beacon["beacon"] = int(number)
                    data_by_beacon['temperature_c'] = data_by_beacon[['T_CO','T_NO2']].mean(axis=1)
                    data_by_beacon.rename(columns={"Timestamp":"timestamp","TVOC":"tvoc","Lux":"lux","NO2":"no2","CO":"co","CO2":"co2",
                                    "PM_N_1":"pm1_number","PM_N_2p5":"pm2p5_number","PM_N_10":"pm10_number",
                                    "PM_C_1":"pm1_mass","PM_C_2p5":"pm2p5_mass","PM_C_10":"pm10_mass"},inplace=True)
                    # correcting
                    try:
                        for var in self.linear_model.keys():
                            data_by_beacon[var] = data_by_beacon[var] * self.linear_model[var].loc[beacon,"coefficient"] + self.linear_model[var].loc[beacon,"constant"]
                    except AttributeError:
                        print("No correction model set")

                    data = data.append(data_by_beacon)
            except FileNotFoundError:
                print(f"No files found for beacon {beacon}.")
                
        #data['temperature_c'] = data[['T_CO','T_NO2']].mean(axis=1)
        data['rh'] = data[['RH_CO','RH_NO2']].mean(axis=1)
        data.drop(["eCO2","Visible","Infrared","Relative Humidity","PM_N_0p5","T_CO","T_NO2","RH_CO","RH_NO2"],axis="columns",inplace=True)
        data = data[[column for column in data.columns if "4" not in column]]
        data.reset_index(inplace=True)
        data.rename(columns={"Timestamp":"timestamp","TVOC":"tvoc","Lux":"lux","NO2":"no2","CO":"co","CO2":"co2",
                                    "PM_N_1":"pm1_number","PM_N_2p5":"pm2p5_number","PM_N_10":"pm10_number",
                                    "PM_C_1":"pm1_mass","PM_C_2p5":"pm2p5_mass","PM_C_10":"pm10_mass","Temperature [C]":"temperature_c_internal"},inplace=True)
        data["co"] /= 1000
        self.data = data

    def set_correction_model(self):
        """
        Sets the class correction models
        """
        # Beacon Attributes
        self.linear_model = {}
        for file in os.listdir(f"{self.data_dir}/interim/"):
            file_info = file.split("-")
            if len(file_info) == 3:
                if file_info[1] == "linear_model" and file_info[-1] == self.suffix+".csv":
                    try:
                        self.linear_model[file_info[0]] = pd.read_csv(f'{self.data_dir}/interim/{file}',index_col=0)
                    except FileNotFoundError:
                        print(f"Missing linear model for {file_info[0]}")
                        self.linear_model[file_info[0]] = pd.DataFrame(data={"beacon":np.arange(1,51),"constant":np.zeros(51),"coefficient":np.ones(51)}).set_index("beacon")